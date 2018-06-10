import random
from typing import Mapping, List, Optional, Iterable, Tuple

import numpy as np
import tensorflow as tf

from jack.core.data_structures import QASetting, Answer
from jack.core.input_module import OnlineInputModule
from jack.core.shared_resources import SharedResources
from jack.core.tensorflow import TFModelModule
from jack.core.tensorport import Ports, TensorPort, TensorPortTensors
from jack.readers.classification.util import create_answer_vocab
from jack.util import preprocessing
from jack.util.map import numpify
from jack.util.tf import misc
from jack.util.tf.rnn import fused_birnn
from projects.knowledge_integration.knowledge_store import KnowledgeStore
from projects.knowledge_integration.shared import AssertionMRPorts
from projects.knowledge_integration.tfutil import embedding_refinement


class MultipleChoiceAssertionInputModule(OnlineInputModule[Mapping[str, any]]):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources
        self.__nlp = preprocessing.spacy_nlp()
        self._rng = random.Random(123)

    @property
    def training_ports(self):
        return [Ports.Target.target_index]

    @property
    def output_ports(self):
        return [AssertionMRPorts.question_length, AssertionMRPorts.support_length,
                # char
                AssertionMRPorts.word_chars, AssertionMRPorts.word_char_length,
                AssertionMRPorts.question, AssertionMRPorts.support,
                # optional, only during training
                AssertionMRPorts.is_eval,
                # for assertionss
                AssertionMRPorts.word_embeddings,
                AssertionMRPorts.assertion_lengths,
                AssertionMRPorts.assertion2question,
                AssertionMRPorts.assertions,
                AssertionMRPorts.question_arg_span,
                AssertionMRPorts.assertion2question_arg_span,
                AssertionMRPorts.support_arg_span,
                AssertionMRPorts.assertion2support_arg_span,
                AssertionMRPorts.word2lemma]

    def preprocess(self, questions: List[QASetting], answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) -> List[Mapping[str, any]]:
        preprocessed = list()
        for i, qa in enumerate(questions):
            tokens, _, length, lemmas, _ = preprocessing.nlp_preprocess(
                qa.question, self.shared_resources.vocab, lowercase=True, with_lemmas=True, use_spacy=True)
            s_tokens, _, s_length, s_lemmas, _ = preprocessing.nlp_preprocess(
                qa.support[0], self.shared_resources.vocab, lowercase=True, with_lemmas=True, use_spacy=True)

            preprocessed.append({
                'support_tokens': s_tokens,
                'support_lemmas': s_lemmas,
                'support_lengths': s_length,
                'question_tokens': tokens,
                'question_lemmas': lemmas,
                'question_lengths': length,
                'ids': i,
            })
            if answers is not None:
                preprocessed[-1]["answers"] = self.shared_resources.answer_vocab(answers[i][0].text)

        return preprocessed

    def create_batch(self, annotations: List[Mapping[str, any]], is_eval: bool, with_answers: bool):
        support_lengths = list()
        question_lengths = list()

        ass_lengths = []
        ass2question = []
        ass2unique = []
        lemma2idx = dict()
        answer_labels = []
        question_arg_span = []
        support_arg_span = []
        assertions2question_arg_span = []
        assertions2support_arg_span = []

        question_arg_span_idx = dict()
        support_arg_span_idx = dict()

        word_chars, word_lengths, tokens, vocab, rev_vocab = \
            preprocessing.unique_words_with_chars(
                [a["question_tokens"] for a in annotations] + [a["support_tokens"] for a in annotations],
                self.char_vocab)
        question, support = tokens[:len(annotations)], tokens[len(annotations):]

        word2lemma = [None] * len(rev_vocab)

        # we have to create batches here and cannot precompute them because of the batch-specific wiq feature
        for i, annot in enumerate(annotations):
            support_lengths.append(annot['support_lengths'])
            question_lengths.append(annot['question_lengths'])

            if "answers" in annot:
                answer_labels.append(annot["answers"])

            # collect uniq lemmas:
            for k, l in enumerate(annot['question_lemmas']):
                if l not in lemma2idx:
                    lemma2idx[l] = len(lemma2idx)
                word2lemma[question[i][k]] = lemma2idx[l]
            for k, l in enumerate(annot['support_lemmas']):
                if l not in lemma2idx:
                    lemma2idx[l] = len(lemma2idx)
                word2lemma[support[i][k]] = lemma2idx[l]

            assertions, assertion_args = self._knowledge_store.get_connecting_assertion_keys(
                annot['question_lemmas'], annot['support_lemmas'], self._sources)

            sorted_assertionss = sorted(assertions.items(), key=lambda x: -x[1])
            added_assertionss = set()
            for key, _ in sorted_assertionss:
                if len(added_assertionss) == self._limit:
                    break
                a = self.__nlp(self._knowledge_store.get_assertion(key))
                a_lemma = " ".join(t.lemma_ for t in a)
                if a_lemma in added_assertionss:
                    continue
                else:
                    added_assertionss.add(a_lemma)
                ass2question.append(i)
                ass_lengths.append(len(a))
                q_arg_span = assertion_args[key][0]
                q_arg_span = (i, q_arg_span[0], q_arg_span[1])
                s_arg_span = assertion_args[key][1]
                s_arg_span = (i, s_arg_span[0], s_arg_span[1])
                if q_arg_span not in question_arg_span_idx:
                    question_arg_span_idx[q_arg_span] = len(question_arg_span)
                    question_arg_span.append(assertion_args[key][0])
                if s_arg_span not in support_arg_span_idx:
                    support_arg_span_idx[s_arg_span] = len(support_arg_span)
                    support_arg_span.append(assertion_args[key][1])
                assertions2question_arg_span.append(question_arg_span_idx[q_arg_span])
                assertions2support_arg_span.append(support_arg_span_idx[s_arg_span])

                u_ass = []
                for t in a:
                    w = t.orth_
                    if w not in vocab:
                        vocab[w] = len(vocab)
                        word_lengths.append(min(len(w), 20))
                        word_chars.append([self.char_vocab.get(c, 0) for c in w[:20]])
                        rev_vocab.append(w)
                        if t.lemma_ not in lemma2idx:
                            lemma2idx[t.lemma_] = len(lemma2idx)
                        word2lemma.append(lemma2idx[t.lemma_])
                    u_ass.append(vocab[w])
                ass2unique.append(u_ass)

        word_embeddings = np.zeros([len(rev_vocab), self.emb_matrix.shape[1]])
        for i, w in enumerate(rev_vocab):
            word_embeddings[i] = self._get_emb(self.shared_resources.vocab(w))

        if not ass2unique:
            ass2unique.append([])
            question_arg_span = support_arg_span = np.zeros([0, 2], dtype=np.int32)

        output = {
            AssertionMRPorts.word_chars: word_chars,
            AssertionMRPorts.word_char_length: word_lengths,
            AssertionMRPorts.question: question,
            AssertionMRPorts.support: support,
            AssertionMRPorts.support_length: support_lengths,
            AssertionMRPorts.question_length: question_lengths,
            AssertionMRPorts.is_eval: is_eval,
            AssertionMRPorts.word_embeddings: word_embeddings,
            AssertionMRPorts.assertion_lengths: ass_lengths,
            AssertionMRPorts.assertion2question: ass2question,
            AssertionMRPorts.assertions: ass2unique,
            AssertionMRPorts.word2lemma: word2lemma,
            AssertionMRPorts.question_arg_span: question_arg_span,
            AssertionMRPorts.support_arg_span: support_arg_span,
            AssertionMRPorts.assertion2question_arg_span: assertions2question_arg_span,
            AssertionMRPorts.assertion2support_arg_span: assertions2support_arg_span,
            '__vocab': vocab,
            '__rev_vocab': rev_vocab,
            '__lemma_vocab': lemma2idx,
        }
        if "answers" in annotations[0]:
            output[Ports.Target.target_index] = [a["answers"] for a in annotations]

        return numpify(output, keys=self.output_ports + self.training_ports)

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        if not self.shared_resources.vocab.frozen:
            self.shared_resources.vocab = preprocessing.fill_vocab(
                (q for q, _ in data), self.shared_resources.vocab, lowercase=True)
            self.shared_resources.vocab.freeze()
        if not hasattr(self.shared_resources, 'answer_vocab') or not self.shared_resources.answer_vocab.frozen:
            self.shared_resources.answer_vocab = create_answer_vocab(answers=(a for _, ass in data for a in ass))
            self.shared_resources.answer_vocab.freeze()
        self.shared_resources.config['answer_size'] = self.shared_resources.config.get(
            'answer_size', len(self.shared_resources.answer_vocab))
        self.shared_resources.char_vocab = {chr(i): i for i in range(256)}

    def setup(self):
        self._knowledge_store = KnowledgeStore(self.shared_resources.config["assertion_dir"])
        self._sources = self.shared_resources.config["assertion_sources"]
        self._limit = self.shared_resources.config.get("assertion_limit", 10)
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config
        self.batch_size = self.config.get("batch_size", 1)
        self.dropout = self.config.get("dropout", 0.0)
        self._rng = random.Random(self.config.get("seed", 123))
        self.emb_matrix = self.vocab.emb.lookup
        self.default_vec = np.zeros([self.vocab.emb_length])
        self.char_vocab = self.shared_resources.char_vocab

    def _get_emb(self, idx):
        if idx < self.emb_matrix.shape[0]:
            return self.emb_matrix[idx]
        else:
            return self.default_vec


class ClassificationAssertionMixin:
    @property
    def input_ports(self) -> List[TensorPort]:
        return [AssertionMRPorts.question_length, AssertionMRPorts.support_length,
                # char embedding inputs
                AssertionMRPorts.word_chars, AssertionMRPorts.word_char_length,
                AssertionMRPorts.question, AssertionMRPorts.support,
                # optional input, provided only during training
                AssertionMRPorts.is_eval,
                # assertions related ports
                AssertionMRPorts.word_embeddings, AssertionMRPorts.assertion_lengths,
                AssertionMRPorts.assertion2question, AssertionMRPorts.assertions,
                AssertionMRPorts.word2lemma]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, Ports.Prediction.candidate_index]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, Ports.Target.target_index]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]


class NLIAssertionModel(ClassificationAssertionMixin, TFModelModule):
    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        question_length = tensors.question_length
        support_length = tensors.support_length
        word_chars = tensors.word_chars
        word_char_length = tensors.word_char_length
        question = tensors.question
        support = tensors.support
        is_eval = tensors.is_eval
        word_embeddings = tensors.word_embeddings
        assertion_length = tensors.assertion_lengths
        assertion2question = tensors.assertion2question
        assertions = tensors.assertions
        word2lemma = tensors.word2lemma

        # Some helpers
        input_size = shared_resources.config["repr_dim_input"]
        size = shared_resources.config["repr_dim"]
        num_classes = shared_resources.config["answer_size"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
        reading_encoder_config = shared_resources.config['reading_module']

        word_embeddings.set_shape([None, input_size])

        reading_sequence = [support, question, assertions]
        reading_sequence_lengths = [support_length, question_length, assertion_length]
        reading_sequence_2_batch = [None, None, assertion2question]

        new_word_embeddings, reading_sequence_offset, _ = embedding_refinement(
            size, word_embeddings, reading_encoder_config,
            reading_sequence, reading_sequence_2_batch, reading_sequence_lengths,
            word2lemma, word_chars, word_char_length, is_eval,
            keep_prob=1.0 - shared_resources.config.get('dropout', 0.0),
            with_char_embeddings=with_char_embeddings, num_chars=len(shared_resources.char_vocab))

        emb_question = tf.nn.embedding_lookup(new_word_embeddings, reading_sequence_offset[1],
                                              name='embedded_question')
        emb_support = tf.nn.embedding_lookup(new_word_embeddings, reading_sequence_offset[0],
                                             name='embedded_support')

        logits = nli_model(size, num_classes, emb_question, question_length, emb_support, support_length)

        return {
            Ports.Prediction.logits: logits,
            Ports.Prediction.candidate_index: tf.argmax(logits, 1)
        }

    def create_training_output(self, shared_resources: SharedResources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        return {
            Ports.loss: tf.losses.sparse_softmax_cross_entropy(logits=tensors.logits, labels=tensors.target_index),
        }


def nli_model(size, num_classes, emb_question, question_length, emb_support, support_length):
    fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(size)
    # [batch, 2*output_dim] -> [batch, num_classes]
    _, q_states = fused_birnn(fused_rnn, emb_question, sequence_length=question_length,
                              dtype=tf.float32, time_major=False, scope="question_rnn")

    outputs, _ = fused_birnn(fused_rnn, emb_support, sequence_length=support_length,
                             dtype=tf.float32, initial_state=q_states, time_major=False, scope="support_rnn")

    # [batch, T, 2 * dim] -> [batch, dim]
    outputs = tf.concat([outputs[0], outputs[1]], axis=2)
    hidden = tf.layers.dense(outputs, size, tf.nn.relu, name="hidden") * tf.expand_dims(
        misc.mask_for_lengths(support_length, max_length=tf.shape(outputs)[1], mask_right=False, value=1.0), 2)
    hidden = tf.reduce_max(hidden, axis=1)
    # [batch, dim] -> [batch, num_classes]
    outputs = tf.layers.dense(hidden, num_classes, name="classification")
    return outputs
