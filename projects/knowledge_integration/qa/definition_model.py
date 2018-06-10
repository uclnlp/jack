import random

import numpy as np
import spacy
import tensorflow as tf

from jack.core import TensorPortWithDefault, TensorPortTensors, TensorPort
from jack.readers.extractive_qa.shared import XQAPorts
from jack.readers.extractive_qa.tensorflow.abstract_model import AbstractXQAModelModule
from jack.readers.extractive_qa.tensorflow.answer_layer import answer_layer
from jack.util.map import numpify
from jack.util.preprocessing import sort_by_tfidf
from jack.util.tf.modular_encoder import modular_encoder
from projects.knowledge_integration.qa.shared import XQAAssertionInputModule
from projects.knowledge_integration.shared import AssertionMRPorts
from projects.knowledge_integration.tfutil import word_with_char_embed, embedding_refinement


class DefinitionPorts:
    definition_lengths = TensorPortWithDefault(np.zeros([0], np.int32), [None],
                                               "definition_lengths", "Length of definition.", "[R]")
    definitions = TensorPortWithDefault(np.zeros([0, 0], np.int32), [None, None], "definitions",
                                        "Represents batch dependent definition word ids.", "[R, L]")
    definition2question = TensorPortWithDefault(np.zeros([0], np.int32), [None], "definition2question",
                                                "Question idx per definition", "[R]")


class XQAAssertionDefinitionInputModule(XQAAssertionInputModule):

    def setup(self):
        super().setup()
        self.use_definitions = True
        self._rng = random.Random(1)

    def set_reader(self, reader):
        self.reader = reader

    @property
    def output_ports(self):
        return super(XQAAssertionDefinitionInputModule, self).output_ports + [
            DefinitionPorts.definitions, DefinitionPorts.definition_lengths, DefinitionPorts.definition2question]

    def create_batch(self, annotations, is_eval, with_answers):
        frac = self.config.get('training_fraction_with_definition', 1.0)
        if not self.use_definitions or (frac < 1.0 and not is_eval and self._rng.random() > frac):
            return super(XQAAssertionDefinitionInputModule, self).create_batch(annotations, is_eval, with_answers)
        batch = super(XQAAssertionDefinitionInputModule, self).create_batch(annotations, True, with_answers)

        lemma_vocab = batch['__lemma_vocab']
        vocab = batch['__vocab']
        rev_vocab = batch['__rev_vocab']
        word_chars = batch[AssertionMRPorts.word_chars].tolist()
        word_lengths = batch[AssertionMRPorts.word_char_length].tolist()
        word2lemma = batch[AssertionMRPorts.word2lemma].tolist()
        support = batch[AssertionMRPorts.support]

        rev_lemma_vocab = {v: k for k, v in lemma_vocab.items()}
        topk = self.config['topk']
        self.reader.model_module.set_topk(topk)
        spans = self.reader.model_module(batch, [XQAPorts.answer_span])[XQAPorts.answer_span]

        definitions = []
        definition_lengths = []
        definition2question = []

        seen_answer_lemmas = None
        for i, s in enumerate(spans):
            j = i // topk
            if i % topk == 0:
                seen_answer_lemmas = set()
            doc_idx_map = [i for i, q_id in enumerate(batch[XQAPorts.support2question]) if q_id == j]
            doc_idx, start, end = s[0], s[1], s[2]
            answer_token_ids = support[doc_idx_map[doc_idx], start:end + 1]
            answer_lemmas = [rev_lemma_vocab[word2lemma[idd]] for idd in answer_token_ids]
            answer_lemma = ' '.join(answer_lemmas)
            if answer_lemma in seen_answer_lemmas:
                continue
            seen_answer_lemmas.add(answer_lemma)
            ks = self._knowledge_store.assertion_keys_for_subject(answer_lemma, resource='wikipedia_firstsent')
            if not ks:
                # remove leading or trailing stop words or non alnum words
                while answer_lemmas and (answer_lemmas[0] in spacy.en.STOP_WORDS or not answer_lemmas[0].isalnum()):
                    answer_lemmas = answer_lemmas[1:]
                while answer_lemmas and (answer_lemmas[-1] in spacy.en.STOP_WORDS or not answer_lemmas[-1].isalnum()):
                    answer_lemmas = answer_lemmas[:-1]
                answer_lemma = ' '.join(answer_lemmas)
                if answer_lemma in seen_answer_lemmas:
                    continue
                seen_answer_lemmas.add(answer_lemma)
                ks = self._knowledge_store.assertion_keys_for_subject(answer_lemma, resource='wikipedia_firstsent')

            defns = [self._nlp(self._knowledge_store.get_assertion(key)) for key in ks]
            if len(defns) > 3:
                indices_scores = sort_by_tfidf(
                    ' '.join(annotations[j].question_lemmas + annotations[j].support_lemmas[doc_idx]),
                    [' '.join(t.lemma_ for t in d) for d in defns])
                # only select the top 3 definition with best match to the support and question
                defns = [defns[i] for i, _ in indices_scores[:3]]

            for defn in defns:
                definition_lengths.append(len(defn))
                definition2question.append(j)
                defn_ids = []
                for t in defn:
                    w = t.orth_
                    if w not in vocab:
                        vocab[w] = len(vocab)
                        word_lengths.append(min(len(w), 20))
                        word_chars.append([self.char_vocab.get(c, 0) for c in w[:20]])
                        rev_vocab.append(w)
                        if t.lemma_ not in lemma_vocab:
                            lemma_vocab[t.lemma_] = len(lemma_vocab)
                        word2lemma.append(lemma_vocab[t.lemma_])
                    defn_ids.append(vocab[w])
                definitions.append(defn_ids)

        batch[DefinitionPorts.definitions] = definitions
        batch[DefinitionPorts.definition_lengths] = definition_lengths
        batch[DefinitionPorts.definition2question] = definition2question
        batch[AssertionMRPorts.word_chars] = word_chars
        batch[AssertionMRPorts.word_char_length] = word_lengths
        batch[AssertionMRPorts.word2lemma] = word2lemma
        batch[AssertionMRPorts.is_eval] = is_eval

        word_embeddings = np.zeros([len(rev_vocab), self.emb_matrix.shape[1]])
        for i, w in enumerate(rev_vocab):
            word_embeddings[i] = self._get_emb(self.vocab(w))

        batch[AssertionMRPorts.word_embeddings] = word_embeddings

        return numpify(batch, keys=[
            DefinitionPorts.definitions, DefinitionPorts.definition_lengths, DefinitionPorts.definition2question,
            AssertionMRPorts.word_chars, AssertionMRPorts.word_char_length, AssertionMRPorts.word2lemma])


class ModularAssertionDefinitionQAModel(AbstractXQAModelModule):
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
                    XQAPorts.answer2support_training,
                    DefinitionPorts.definitions,
                    DefinitionPorts.definition_lengths,
                    DefinitionPorts.definition2question]

    @property
    def input_ports(self):
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
        definition_lengths = tensors.definition_lengths
        definition2question = tensors.definition2question
        definitions = tensors.definitions
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
                reading_sequence = [support, question, assertions, definitions]
                reading_sequence_lengths = [support_length, question_length, assertion_lengths, definition_lengths]
                reading_sequence_to_batch = [support2question, None, assertion2question, definition2question]
            else:
                reading_sequence = [support, question, definitions]
                reading_sequence_lengths = [support_length, question_length, definition_lengths]
                reading_sequence_to_batch = [support2question, None, definition2question]

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
                  'word_in_question': tf.expand_dims(tensors.word_in_question, 2)}
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

        span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))
