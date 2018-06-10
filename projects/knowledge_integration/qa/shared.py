import random
from typing import *

import numpy as np
import tensorflow as tf

from jack.core import TensorPortTensors, TensorPort
from jack.readers.extractive_qa.shared import XQAInputModule, XQAPorts
from jack.readers.extractive_qa.tensorflow.abstract_model import AbstractXQAModelModule
from jack.readers.extractive_qa.tensorflow.answer_layer import answer_layer
from jack.readers.extractive_qa.util import prepare_data
from jack.util import preprocessing
from jack.util.map import numpify
from jack.util.preprocessing import sort_by_tfidf
from jack.util.tf.modular_encoder import modular_encoder
from projects.knowledge_integration.knowledge_store import KnowledgeStore
from projects.knowledge_integration.shared import AssertionMRPorts
from projects.knowledge_integration.tfutil import embedding_refinement, word_with_char_embed

XQAAssertionAnnotation = NamedTuple('XQAAssertionAnnotation', [
    ('question_tokens', List[str]),
    ('question_lemmas', List[str]),
    ('question_ids', List[int]),
    ('question_length', int),
    ('support_tokens', List[str]),
    ('support_lemmas', List[str]),
    ('support_ids', List[int]),
    ('support_length', int),
    ('word_in_question', List[float]),
    ('token_offsets', List[int]),
    ('answer_spans', Optional[List[Tuple[int, int]]]),
    ('selected_supports', List[int]),
])


class XQAAssertionInputModule(XQAInputModule):
    _output_ports = [AssertionMRPorts.question_length, AssertionMRPorts.support_length,
                     # char
                     AssertionMRPorts.word_chars, AssertionMRPorts.word_char_length,
                     AssertionMRPorts.question, AssertionMRPorts.support,
                     # optional, only during training
                     AssertionMRPorts.is_eval,
                     # for assertions
                     AssertionMRPorts.word_embeddings,
                     AssertionMRPorts.assertion_lengths,
                     AssertionMRPorts.assertion2question,
                     AssertionMRPorts.assertions,
                     AssertionMRPorts.question_arg_span,
                     AssertionMRPorts.assertion2question_arg_span,
                     AssertionMRPorts.support_arg_span,
                     AssertionMRPorts.assertion2support_arg_span,
                     AssertionMRPorts.word2lemma,
                     XQAPorts.word_in_question,
                     XQAPorts.support2question,
                     # optional, only during training
                     XQAPorts.answer2support_training, XQAPorts.correct_start,
                     # for output module
                     XQAPorts.token_offsets, XQAPorts.selected_support]

    def __init__(self, shared_resources):
        super(XQAAssertionInputModule, self).__init__(shared_resources)
        self._nlp = preprocessing.spacy_nlp()
        self._rng = random.Random(123)

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

    @property
    def output_ports(self):
        return self._output_ports

    def preprocess_instance(self, question, answers=None):
        has_answers = answers is not None

        q_tokenized, q_ids, q_lemmas, q_length, s_tokenized, s_ids, s_lemmas, s_length, \
        word_in_question, token_offsets, answer_spans = prepare_data(
            question, answers, self.vocab, self.config.get("lowercase", False),
            with_answers=has_answers, max_support_length=self.config.get("max_support_length", None),
            spacy_nlp=True, with_lemmas=True)

        max_num_support = self.config.get("max_num_support", len(question.support))  # take all per default

        # take max supports by TF-IDF (we subsample to max_num_support in create batch)
        # following https://arxiv.org/pdf/1710.10723.pdf
        if len(question.support) > 1:
            scores = sort_by_tfidf(' '.join(q_tokenized), [' '.join(s) for s in s_tokenized])
            selected_supports = [s_idx for s_idx, _ in scores[:max_num_support]]
            s_tokenized = [s_tokenized[s_idx] for s_idx in selected_supports]
            s_lemmas = [s_lemmas[s_idx] for s_idx in selected_supports]
            s_ids = [s_ids[s_idx] for s_idx in selected_supports]
            s_length = [s_length[s_idx] for s_idx in selected_supports]
            word_in_question = [word_in_question[s_idx] for s_idx in selected_supports]
            token_offsets = [token_offsets[s_idx] for s_idx in selected_supports]
            answer_spans = [answer_spans[s_idx] for s_idx in selected_supports]
        else:
            selected_supports = list(range(len(question.support)))

        return XQAAssertionAnnotation(
            question_tokens=q_tokenized,
            question_lemmas=q_lemmas,
            question_ids=q_ids,
            question_length=q_length,
            support_tokens=s_tokenized,
            support_lemmas=s_lemmas,
            support_ids=s_ids,
            support_length=s_length,
            word_in_question=word_in_question,
            token_offsets=token_offsets,
            answer_spans=answer_spans if has_answers else None,
            selected_supports=selected_supports,
        )

    def create_batch(self, annotations, is_eval: bool, with_answers: bool):
        q_tokenized = [a.question_tokens for a in annotations]
        question_lengths = [a.question_length for a in annotations]

        max_training_support = self.config.get('max_training_support', 2)
        s_tokenized = []
        s_lemmas = []
        support_lengths = []
        wiq = []
        offsets = []
        support2question = []
        # aligns with support2question, used in output module to get correct index to original set of supports
        selected_support = []
        all_spans = []
        for i, a in enumerate(annotations):
            s_lemmas.append([])
            all_spans.append([])
            if len(a.support_tokens) > max_training_support > 0 and not is_eval:
                # sample only 2 paragraphs and take first with double probability (the best) to speed
                # things up. Following https://arxiv.org/pdf/1710.10723.pdf
                is_done = False
                any_answer = any(a.answer_spans)
                # sample until there is at least one possible answer (if any)
                while not is_done:
                    selected = self._rng.sample(range(0, len(a.support_tokens) + 1), max_training_support + 1)
                    if 0 in selected and 1 in selected:
                        selected = [s - 1 for s in selected if s > 0]
                    else:
                        selected = [max(0, s - 1) for s in selected[:max_training_support]]
                    is_done = not any_answer or any(a.answer_spans[s] for s in selected)
                selected = set(max(0, s - 1) for s in selected)
            else:
                selected = set(range(len(a.support_tokens)))
            for s in selected:
                s_tokenized.append(a.support_tokens[s])
                s_lemmas[-1].append(a.support_lemmas[s])
                support_lengths.append(a.support_length[s])
                wiq.append(a.word_in_question[s])
                offsets.append(a.token_offsets[s])
                selected_support.append(a.selected_supports[s])
                support2question.append(i)
                if with_answers:
                    all_spans[-1].append(a.answer_spans[s])

        word_chars, word_lengths, word_ids, vocab, rev_vocab = \
            preprocessing.unique_words_with_chars(q_tokenized + s_tokenized, self.char_vocab)

        question = word_ids[:len(q_tokenized)]
        support = word_ids[len(q_tokenized):]

        ass_lengths = []
        ass2question = []
        ass2unique = []
        lemma2idx = dict()
        question_arg_span = []
        support_arg_span = []
        assertion2question_arg_span = []
        assertion2support_arg_span = []
        question_arg_span_idx = dict()
        support_arg_span_idx = dict()

        word2lemma = [None] * len(rev_vocab)

        heuristic = self.config.get('heuristic', 'pair')
        s_offset = 0
        for i, annot in enumerate(annotations):
            # collect uniq lemmas:
            for k, l in enumerate(annot.question_lemmas):
                if l not in lemma2idx:
                    lemma2idx[l] = len(lemma2idx)
                word2lemma[question[i][k]] = lemma2idx[l]
            for k, ls in enumerate(s_lemmas[i]):
                for k2, l in enumerate(ls):
                    if l not in lemma2idx:
                        lemma2idx[l] = len(lemma2idx)
                    word2lemma[support[s_offset + k][k2]] = lemma2idx[l]

            if self._limit == 0:
                s_offset += len(s_lemmas[i])
                continue

            if heuristic == 'pair':
                assertions, assertion_args = self._knowledge_store.get_connecting_assertion_keys(
                    annot.question_lemmas, [l for ls in s_lemmas[i] for l in ls], self._sources)
            elif heuristic == 'tfidf':
                assertions, assertion_args = self._knowledge_store.get_assertion_keys(
                    [l for ls in s_lemmas[i] for l in ls], self._sources)
                assertions = list(assertions.keys())
                assertion_strings = [self._knowledge_store.get_assertion(key) for key in assertions]
                scores = sort_by_tfidf(' '.join(annot.question_tokens), assertion_strings)
                assertions = {assertions[i]: s for i, s in scores}

            sorted_assertions = sorted(assertions.items(), key=lambda x: -x[1])
            added_assertions = set()
            for key, _ in sorted_assertions:
                if len(added_assertions) == self._limit:
                    break
                a = self._nlp(self._knowledge_store.get_assertion(key, cache=True))
                a_lemma = " ".join(t.lemma_ for t in a)
                if a_lemma in added_assertions:
                    continue
                else:
                    added_assertions.add(a_lemma)
                ass2question.append(i)
                ass_lengths.append(len(a))
                if heuristic == 'pair':
                    q_arg_span = assertion_args[key][0]
                    q_arg_span = (i, q_arg_span[0], q_arg_span[1])
                    s_arg_start, s_arg_end = assertion_args[key][1]
                    doc_idx = 0
                    for ls in s_lemmas[i]:
                        if s_arg_start < len(ls):
                            break
                        else:
                            doc_idx += 1
                            s_arg_start -= len(ls)
                            s_arg_end -= len(ls)
                    s_arg_span = (s_offset + doc_idx, s_arg_start, s_arg_end)
                    if q_arg_span not in question_arg_span_idx:
                        question_arg_span_idx[q_arg_span] = len(question_arg_span)
                        question_arg_span.append(assertion_args[key][0])
                    if s_arg_span not in support_arg_span_idx:
                        support_arg_span_idx[s_arg_span] = len(support_arg_span)
                        support_arg_span.append(assertion_args[key][1])
                    assertion2question_arg_span.append(question_arg_span_idx[q_arg_span])
                    assertion2support_arg_span.append(support_arg_span_idx[s_arg_span])

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

            s_offset += len(s_lemmas[i])

        word_embeddings = np.zeros([len(rev_vocab), self.emb_matrix.shape[1]])
        for i, w in enumerate(rev_vocab):
            word_embeddings[i] = self._get_emb(self.vocab(w))

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
            AssertionMRPorts.assertion2question_arg_span: assertion2question_arg_span,
            AssertionMRPorts.assertion2support_arg_span: assertion2support_arg_span,
            XQAPorts.word_in_question: wiq,
            XQAPorts.support2question: support2question,
            XQAPorts.token_offsets: offsets,
            XQAPorts.selected_support: selected_support,
            '__vocab': vocab,
            '__rev_vocab': rev_vocab,
            '__lemma_vocab': lemma2idx,
        }

        if with_answers:
            spans = [s for a in all_spans for spans_per_support in a for s in spans_per_support]
            span2support = []
            support_idx = 0
            for a in all_spans:
                for spans_per_support in a:
                    span2support.extend([support_idx] * len(spans_per_support))
                    support_idx += 1
            output.update({
                XQAPorts.answer_span_target: [span for span in spans] if spans else np.zeros([0, 2], np.int32),
                XQAPorts.correct_start: [] if is_eval else [span[0] for span in spans],
                XQAPorts.answer2support_training: span2support,
            })

        # we can only numpify in here, because bucketing is not possible prior
        batch = numpify(output, keys=self.output_ports + self.training_ports)
        return batch


class ModularAssertionQAModel(AbstractXQAModelModule):
    _input_ports = [AssertionMRPorts.question_length, AssertionMRPorts.support_length,
                    # char
                    AssertionMRPorts.word_chars, AssertionMRPorts.word_char_length,
                    AssertionMRPorts.question, AssertionMRPorts.support,
                    # optional, only during training
                    AssertionMRPorts.is_eval,
                    # for assertions
                    AssertionMRPorts.word_embeddings,
                    AssertionMRPorts.assertion_lengths,
                    AssertionMRPorts.assertion2question,
                    AssertionMRPorts.assertions,
                    AssertionMRPorts.word2lemma,
                    XQAPorts.word_in_question,
                    XQAPorts.support2question,
                    XQAPorts.correct_start,
                    XQAPorts.answer2support_training]

    @property
    def input_ports(self) -> Sequence[TensorPort]:
        return self._input_ports

    def set_topk(self, k):
        self._topk_assign(k)

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        question_length = tensors.question_length
        support_length = tensors.support_length
        support2question = tensors.support2question
        word_chars = tensors.word_chars
        word_char_length = tensors.word_char_length
        question = tensors.question
        support = tensors.support
        is_eval = tensors.is_eval
        word_embeddings = tensors.word_embeddings
        assertion_lengths = tensors.assertion_lengths
        assertion2question = tensors.assertion2question
        assertions = tensors.assertions
        word2lemma = tensors.word2lemma

        model = shared_resources.config['model']
        repr_dim = shared_resources.config['repr_dim']
        input_size = shared_resources.config["repr_dim_input"]
        dropout = shared_resources.config.get("dropout", 0.0)
        size = shared_resources.config["repr_dim"]
        with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)

        word_embeddings.set_shape([None, input_size])

        if shared_resources.config.get('no_reading', False):
            new_word_embeddings = tf.layers.dense(word_embeddings, size, activation=tf.nn.relu,
                                                  name="embeddings_projection")
            if with_char_embeddings:
                new_word_embeddings = word_with_char_embed(
                    size, new_word_embeddings, tensors.word_chars, tensors.word_char_length,
                    len(shared_resources.char_vocab))
            keep_prob = 1.0 - dropout
            if keep_prob < 1.0:
                new_word_embeddings = tf.cond(is_eval,
                                              lambda: new_word_embeddings,
                                              lambda: tf.nn.dropout(new_word_embeddings, keep_prob, [1, size]))
            reading_sequence_offset = [support, question, assertions]
        else:
            if shared_resources.config.get("assertion_limit", 0) > 0:
                reading_sequence = [support, question, assertions]
                reading_sequence_lengths = [support_length, question_length, assertion_lengths]
                reading_sequence_to_batch = [support2question, None, assertion2question]
            else:
                reading_sequence = [support, question]
                reading_sequence_lengths = [support_length, question_length]
                reading_sequence_to_batch = [support2question, None]

            reading_encoder_config = shared_resources.config['reading_module']
            new_word_embeddings, reading_sequence_offset, _ = embedding_refinement(
                size, word_embeddings, reading_encoder_config,
                reading_sequence, reading_sequence_to_batch, reading_sequence_lengths,
                word2lemma, word_chars, word_char_length, is_eval,
                keep_prob=1.0 - shared_resources.config.get('dropout', 0.0),
                with_char_embeddings=with_char_embeddings, num_chars=len(shared_resources.char_vocab))

        emb_question = tf.nn.embedding_lookup(new_word_embeddings, reading_sequence_offset[1],
                                              name='embedded_question')
        emb_support = tf.nn.embedding_lookup(new_word_embeddings, reading_sequence_offset[0],
                                             name='embedded_support')

        inputs = {'question': emb_question, 'support': emb_support,
                  'word_in_question': tf.expand_dims(tensors.word_in_question, 2),
                  'question_ones': tf.expand_dims(tf.ones(tf.shape(emb_question)[:2], tf.float32), 2)}
        inputs_length = {'question': question_length, 'support': support_length,
                         'word_in_question': support_length}
        inputs_mapping = {'question': None, 'support': support2question}

        encoder_config = model['encoder_layer']

        encoded, lengths, mappings = modular_encoder(
            encoder_config, inputs, inputs_length, inputs_mapping, repr_dim, dropout, tensors.is_eval)

        with tf.variable_scope('answer_layer'):
            answer_layer_config = model['answer_layer']
            encoded_question = encoded[answer_layer_config.get('question', 'question')]
            encoded_support = encoded[answer_layer_config.get('support', 'support')]

            if 'repr_dim' not in answer_layer_config:
                answer_layer_config['repr_dim'] = repr_dim
            if 'max_span_size' not in answer_layer_config:
                answer_layer_config['max_span_size'] = shared_resources.config.get('max_span_size', 16)
            topk = tf.get_variable(
                'topk', initializer=shared_resources.config.get('topk', 1), dtype=tf.int32, trainable=False)
            topk_p = tf.placeholder(tf.int32, [], 'topk_setter')
            topk_assign = topk.assign(topk_p)
            self._topk_assign = lambda k: self.tf_session.run(topk_assign, {topk_p: k})

            start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                answer_layer(encoded_question, lengths[answer_layer_config.get('question', 'question')],
                             encoded_support, lengths[answer_layer_config.get('support', 'support')],
                             mappings[answer_layer_config.get('support', 'support')],
                             tensors.answer2support, tensors.is_eval,
                             tensors.correct_start, topk=topk, **answer_layer_config)

        span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))
