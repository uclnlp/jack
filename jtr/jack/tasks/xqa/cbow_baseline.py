"""
This file contains CBOW baseline specific modules and ports
"""

import random

import spacy

from jtr.jack.core import *
from jtr.jack.fun import simple_model_module, no_shared_resources
from jtr.jack.tasks.xqa.fastqa import XQAPorts
from jtr.jack.tasks.xqa.util import char_vocab_from_vocab, prepare_data, unique_words_with_chars
from jtr.jack.tf_fun.dropout import fixed_dropout
from jtr.jack.tf_fun.embedding import conv_char_embedding_alt
from jtr.jack.tf_fun.xqa import xqa_min_crossentropy_span_loss
from jtr.preprocess.batch import GeneratorWithRestart
from jtr.preprocess.map import numpify
from jtr.util import tfutil

_max_span_size = 10


class CBOWXqaPorts:
    answer_type_span = TensorPort(tf.int32, [None, 2], "question_chars",
                                  "Represents pre-computed answer-type span in question",
                                  "[Q, 2]")

    span_candidates = TensorPort(tf.int32, [None, None, 2], "spans_for_prediction", "all spans for which are predicted",
                                 "[Q, SP, 2]")

    span_scores = TensorPort(tf.float32, [None], "span_scores for each question", "span scores",
                             "[Q, SP]")


class CBOWXqaInputModule(InputModule):
    def __init__(self, shared_vocab_config):
        assert isinstance(shared_vocab_config, SharedVocabAndConfig), \
            "shared_resources for FastQAInputModule must be an instance of SharedVocabAndConfig"
        self.shared_vocab_config = shared_vocab_config
        self.__nlp = spacy.load('en', parser=False)

    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        # create character vocab + word lengths + char ids per word
        self.shared_vocab_config.config["char_vocab"] = char_vocab_from_vocab(self.shared_vocab_config.vocab)
        # Assumes that vocab and embeddings are given during creation
        self.setup()

    def setup(self):
        self.vocab = self.shared_vocab_config.vocab
        self.config = self.shared_vocab_config.config
        self.batch_size = self.config.get("batch_size", 1)
        self.dropout = self.config.get("dropout", 1)
        self._rng = random.Random(self.config.get("seed", 123))
        self.emb_matrix = self.vocab.emb.lookup
        self.default_vec = np.zeros([self.vocab.emb_length])
        self.char_vocab = self.shared_vocab_config.config["char_vocab"]

    def _get_emb(self, idx):
        if idx < self.emb_matrix.shape[0]:
            return self.emb_matrix[idx]
        else:
            return self.default_vec

    def __extract_answertype_span(self, tokens):
        question = " ".join(tokens)
        doc = self.__nlp(question)
        start_id = -1
        end_id = -1
        for t in doc:
            if t.orth_.startswith("wh") or t.orth_ == "how":
                start_id = t.i
            if start_id >= 0 and t.tag_.startswith("V"):
                if doc[0].orth_.lower() == "what" and t.lemma_ == "be" and t.i == 1:
                    for i in range(t.i + 1, len(doc)):
                        if doc[i].pos_.startswith("N") == 1:
                            j = i + 1
                            while j < len(doc) and doc[j].pos_.startswith("N"):
                                j += 1
                            start_id = t.i + 1
                            end_id = j - 1
                            break
                break
            else:
                end_id = t.i

        if start_id < 0:
            start_id = 0
            end_id = len(doc) - 1
        if end_id < start_id:
            end_id = len(doc) - 1

        return [start_id, end_id]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [XQAPorts.emb_question, XQAPorts.question_length,
                XQAPorts.emb_support, XQAPorts.support_length,
                # char
                XQAPorts.unique_word_chars, XQAPorts.unique_word_char_length,
                XQAPorts.question_words2unique, XQAPorts.support_words2unique,
                # features
                XQAPorts.word_in_question,
                # optional, only during training
                XQAPorts.correct_start_training, XQAPorts.answer2question_training,
                XQAPorts.keep_prob, XQAPorts.is_eval,
                # for output module
                XQAPorts.token_char_offsets,
                CBOWXqaPorts.answer_type_span]

    @property
    def training_ports(self) -> List[TensorPort]:
        return [XQAPorts.answer_span, XQAPorts.answer2question]

    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]], is_eval: bool) \
            -> Iterable[Mapping[TensorPort, np.ndarray]]:
        q_tokenized, q_ids, q_lengths, s_tokenized, s_ids, s_lengths, \
        word_in_question, token_offsets, answer_spans = \
            prepare_data(dataset, self.vocab, self.config.get("lowercase", False), with_answers=True,
                         wiq_contentword=True, with_spacy=True,
                         max_support_length=self.config.get("max_support_length", None))

        not_allowed = set(i for i, ss in enumerate(answer_spans)
                          if not is_eval and all(s[1] - s[0] > _max_span_size for s in ss))

        answertype_spans = []
        for qs in q_tokenized:
            answertype_spans.append(self.__extract_answertype_span(qs))

        emb_supports = np.zeros([self.batch_size, max(s_lengths), self.emb_matrix.shape[1]])
        emb_questions = np.zeros([self.batch_size, max(q_lengths), self.emb_matrix.shape[1]])

        def batch_generator():
            todo = list(i for i in range(len(q_ids)) if is_eval or i not in not_allowed)
            if not is_eval:
                self._rng.shuffle(todo)
            while todo:
                support_lengths = list()
                question_lengths = list()
                wiq = list()
                spans = list()
                span2question = []
                offsets = []
                at_spans = []

                unique_words, unique_word_lengths, question2unique, support2unique = \
                    unique_words_with_chars(q_tokenized, s_tokenized, self.char_vocab, todo[:self.batch_size])

                # we have to create batches here and cannot precompute them because of the batch-specific wiq feature
                for i, j in enumerate(todo[:self.batch_size]):
                    support = s_ids[j]
                    for k in range(len(support)):
                        emb_supports[i, k] = self._get_emb(support[k])
                    question = q_ids[j]
                    for k in range(len(question)):
                        emb_questions[i, k] = self._get_emb(question[k])
                    support_lengths.append(s_lengths[j])
                    question_lengths.append(q_lengths[j])
                    aps = [s for s in answer_spans[j] if s[1] - s[0] <= _max_span_size or is_eval]
                    spans.extend(aps)
                    span2question.extend(i for _ in aps)
                    wiq.append(word_in_question[j])
                    offsets.append(token_offsets[j])
                    at_spans.append(answertype_spans[j])

                batch_size = len(question_lengths)
                output = {
                    XQAPorts.unique_word_chars: unique_words,
                    XQAPorts.unique_word_char_length: unique_word_lengths,
                    XQAPorts.question_words2unique: question2unique,
                    XQAPorts.support_words2unique: support2unique,
                    XQAPorts.emb_support: emb_supports[:batch_size, :max(support_lengths), :],
                    XQAPorts.support_length: support_lengths,
                    XQAPorts.emb_question: emb_questions[:batch_size, :max(question_lengths), :],
                    XQAPorts.question_length: question_lengths,
                    XQAPorts.word_in_question: wiq,
                    XQAPorts.answer_span: spans,
                    XQAPorts.correct_start_training: [] if is_eval else [s[0] for s in spans],
                    XQAPorts.answer2question: span2question,
                    XQAPorts.answer2question_training: [] if is_eval else span2question,
                    XQAPorts.keep_prob: 1.0 if is_eval else 1 - self.dropout,
                    XQAPorts.is_eval: is_eval,
                    XQAPorts.token_char_offsets: offsets,
                    CBOWXqaPorts.answer_type_span: at_spans
                }

                # we can only numpify in here, because bucketing is not possible prior
                batch = numpify(output, keys=[XQAPorts.unique_word_chars,
                                              XQAPorts.question_words2unique, XQAPorts.support_words2unique,
                                              XQAPorts.word_in_question, XQAPorts.token_char_offsets])
                todo = todo[self.batch_size:]
                yield batch

        return GeneratorWithRestart(batch_generator)

    def __call__(self, qa_settings: List[QASetting]) -> Mapping[TensorPort, np.ndarray]:
        q_tokenized, q_ids, q_lengths, s_tokenized, s_ids, s_lengths, \
        word_in_question, token_offsets, answer_spans = \
            prepare_data(qa_settings, self.vocab, self.config.get("lowercase", False), with_answers=False,
                         wiq_contentword=True, with_spacy=True)

        answertype_spans = []
        for qs in q_tokenized:
            answertype_spans.append(self.__extract_answertype_span(qs))

        unique_words, unique_word_lengths, question2unique, support2unique = \
            unique_words_with_chars(q_tokenized, s_tokenized, self.char_vocab)

        batch_size = len(qa_settings)
        emb_supports = np.zeros([batch_size, max(s_lengths), self.emb_matrix.shape[1]])
        emb_questions = np.zeros([batch_size, max(q_lengths), self.emb_matrix.shape[1]])

        for i, q in enumerate(q_ids):
            for k, v in enumerate(s_ids[i]):
                emb_supports[i, k] = self._get_emb(v)
            for k, v in enumerate(q):
                emb_questions[i, k] = self._get_emb(v)

        output = {
            XQAPorts.unique_word_chars: unique_words,
            XQAPorts.unique_word_char_length: unique_word_lengths,
            XQAPorts.question_words2unique: question2unique,
            XQAPorts.support_words2unique: support2unique,
            XQAPorts.emb_support: emb_supports,
            XQAPorts.support_length: s_lengths,
            XQAPorts.emb_question: emb_questions,
            XQAPorts.question_length: q_lengths,
            XQAPorts.word_in_question: word_in_question,
            XQAPorts.token_char_offsets: token_offsets,
            CBOWXqaPorts.answer_type_span: answertype_spans
        }

        output = numpify(output, keys=[XQAPorts.unique_word_chars, XQAPorts.question_words2unique,
                                       XQAPorts.support_words2unique, XQAPorts.word_in_question,
                                       XQAPorts.token_char_offsets])

        return output


cbow_xqa_like_model_module_factory = simple_model_module(
    input_ports=[XQAPorts.emb_question, XQAPorts.question_length,
                 XQAPorts.emb_support, XQAPorts.support_length,
                 # char embedding inputs
                 XQAPorts.unique_word_chars, XQAPorts.unique_word_char_length,
                 XQAPorts.question_words2unique, XQAPorts.support_words2unique,
                 # feature input
                 XQAPorts.word_in_question,
                 # optional input, provided only during training
                 XQAPorts.correct_start_training, XQAPorts.answer2question_training,
                 XQAPorts.keep_prob, XQAPorts.is_eval,
                 CBOWXqaPorts.answer_type_span],
    output_ports=[CBOWXqaPorts.span_scores, CBOWXqaPorts.span_candidates,
                  XQAPorts.span_prediction],
    training_input_ports=[CBOWXqaPorts.span_scores, CBOWXqaPorts.span_candidates,
                          XQAPorts.answer_span, XQAPorts.answer2question],
    training_output_ports=[Ports.loss])


def cbow_xqa_like_with_min_crossentropy_loss_factory(shared_resources, f):
    return cbow_xqa_like_model_module_factory(shared_resources, f, no_shared_resources(xqa_min_crossentropy_span_loss))


# Very specialized and therefore not sharable  TF code for fast qa model.
def cbow_xqa_model_module(shared_vocab_config):
    return cbow_xqa_like_with_min_crossentropy_loss_factory(shared_vocab_config, cbow_xqa_model)


def cbow_xqa_model(shared_vocab_config, emb_question, question_length,
                   emb_support, support_length,
                   unique_word_chars, unique_word_char_length,
                   question_words2unique, support_words2unique,
                   word_in_question,
                   correct_start, answer2question, keep_prob, is_eval,
                   answer_type_span):
    """
    cbow_baseline_model model
    Args:
        shared_vocab_config: has at least a field config (dict) with keys "rep_dim", "rep_dim_input"
        emb_question: [Q, L_q, N]
        question_length: [Q]
        emb_support: [Q, L_s, N]
        support_length: [Q]
        unique_word_chars
        unique_word_char_length
        question_words2unique
        support_words2unique
        word_in_question: [Q, L_s]
        correct_start: [A], only during training, window_size.e., is_eval=False
        answer2question: [A], only during training, window_size.e., is_eval=False
        keep_prob: []
        is_eval: []
        answer_type_span: [Q, 2], span within question marking the expected answer type

    Returns:
        start_scores [B, L_s, N], end_scores [B, L_s, N], span_prediction [B, 2]
    """
    with tf.variable_scope("cbow_xqa", initializer=tf.contrib.layers.xavier_initializer()):
        # Some helpers
        batch_size = tf.shape(question_length)[0]
        max_support_length = tf.reduce_max(support_length)
        max_question_length = tf.reduce_max(question_length)

        input_size = shared_vocab_config.config["repr_dim_input"]
        size = shared_vocab_config.config["repr_dim"]
        with_char_embeddings = shared_vocab_config.config.get("with_char_embeddings", False)

        # set shapes for inputs
        emb_question.set_shape([None, None, input_size])
        emb_support.set_shape([None, None, input_size])

        if with_char_embeddings:
            # compute combined embeddings
            [char_emb_question, char_emb_support] = conv_char_embedding_alt(shared_vocab_config.config["char_vocab"],
                                                                            size,
                                                                            unique_word_chars, unique_word_char_length,
                                                                            [question_words2unique,
                                                                             support_words2unique])

            emb_question = tf.concat([emb_question, char_emb_question], 2)
            emb_support = tf.concat([emb_support, char_emb_support], 2)
            input_size += size

            # set shapes for inputs
            emb_question.set_shape([None, None, input_size])
            emb_support.set_shape([None, None, input_size])

        # variational dropout
        dropout_shape = tf.unstack(tf.shape(emb_question))
        dropout_shape[1] = 1

        [emb_question, emb_support] = tf.cond(is_eval,
                                              lambda: [emb_question, emb_support],
                                              lambda: fixed_dropout([emb_question, emb_support],
                                                                    keep_prob, dropout_shape))

        # question encoding
        answer_type_start = tf.squeeze(tf.slice(answer_type_span, [0, 0], [-1, 1]))
        answer_type_end = tf.squeeze(tf.slice(answer_type_span, [0, 1], [-1, -1]))
        answer_type_mask = tfutil.mask_for_lengths(answer_type_start, batch_size, max_question_length, value=1.0) * \
                           tfutil.mask_for_lengths(answer_type_end + 1, batch_size, max_question_length,
                                                   mask_right=False, value=1.0)
        answer_type = tf.reduce_sum(emb_question * tf.expand_dims(answer_type_mask, 2), 1) / \
                      tf.maximum(1.0, tf.reduce_sum(answer_type_mask, 1, keep_dims=True))

        batch_size_range = tf.range(0, batch_size)
        answer_type_start_state = tf.gather_nd(emb_question, tf.stack([batch_size_range, answer_type_start], 1))
        answer_type_end_state = tf.gather_nd(emb_question, tf.stack([batch_size_range, answer_type_end], 1))

        question_rep = tf.concat([answer_type, answer_type_start_state, answer_type_end_state], 1)
        question_rep.set_shape([None, input_size * 3])

        # wiq features
        support_mask = tfutil.mask_for_lengths(support_length, batch_size)
        question_binary_mask = tfutil.mask_for_lengths(question_length, batch_size, mask_right=False, value=1.0)

        v_wiqw = tf.get_variable("v_wiq_w", [1, 1, input_size],
                                 initializer=tf.constant_initializer(1.0))

        wiq_w = tf.matmul(emb_question * v_wiqw, emb_support, adjoint_b=True)
        wiq_w = wiq_w + tf.expand_dims(support_mask, 1)

        wiq_w = tf.reduce_sum(tf.nn.softmax(wiq_w) * tf.expand_dims(question_binary_mask, 2), [1])

        wiq_exp = tf.stack([word_in_question, wiq_w], 2)

        # support span encoding
        spans = [tf.stack([tf.range(0, max_support_length), tf.range(0, max_support_length)], 1)]

        wiq_exp = tf.pad(wiq_exp, [[0, 0], [20, 20], [0, 0]])
        wiq_pooled5 = tf.layers.average_pooling1d(
            tf.slice(wiq_exp, [0, 15, 0], tf.stack([-1, max_support_length + 10, -1])), 5, [1], 'valid')
        wiq_pooled10 = tf.layers.average_pooling1d(
            tf.slice(wiq_exp, [0, 10, 0], tf.stack([-1, max_support_length + 20, -1])), 10, [1], 'valid')
        wiq_pooled20 = tf.layers.average_pooling1d(wiq_exp, 20, [1], 'valid')

        wiqs_left5 = [tf.slice(wiq_pooled5, [0, 0, 0], tf.stack([-1, max_support_length, -1]))]
        wiqs_right5 = [tf.slice(wiq_pooled5, [0, 6, 0], [-1, -1, -1])]
        wiqs_left10 = [tf.slice(wiq_pooled10, [0, 0, 0], tf.stack([-1, max_support_length, -1]))]
        wiqs_right10 = [tf.slice(wiq_pooled10, [0, 11, 0], [-1, -1, -1])]
        wiqs_left20 = [tf.slice(wiq_pooled20, [0, 0, 0], tf.stack([-1, max_support_length, -1]))]
        wiqs_right20 = [tf.slice(wiq_pooled20, [0, 21, 0], [-1, -1, -1])]

        context_window = 5
        padded_support = tf.pad(emb_support, [[0, 0], [context_window, context_window], [0, 0]], "CONSTANT")
        # [B, L + 10 - 4, S]
        emb_support_windows = tf.layers.average_pooling1d(padded_support, 5, [1], "VALID", "channels_last")

        left_context_windows = tf.slice(emb_support_windows, [0, 0, 0],
                                        tf.stack([-1, max_support_length, -1]))
        right_context_windows = tf.slice(emb_support_windows, [0, context_window + 1, 0],
                                         [-1, -1, -1])
        span_rep = [tf.concat([emb_support, emb_support, emb_support, left_context_windows, right_context_windows], 2)]

        for window_size in range(2, _max_span_size + 1):
            start = tf.slice(emb_support, [0, 0, 0], tf.stack([-1, max_support_length - (window_size - 1), -1]))
            end = tf.slice(emb_support, [0, window_size - 1, 0], [-1, -1, -1])
            averagespan = tf.layers.average_pooling1d(emb_support, window_size, [1], "VALID", "channels_last")

            left_context_windows = tf.slice(emb_support_windows, [0, 0, 0],
                                            tf.stack([-1, max_support_length - (window_size - 1), -1]))
            right_context_windows = tf.slice(emb_support_windows, [0, window_size - 1 + context_window + 1, 0],
                                             [-1, -1, -1])

            span_rep.append(tf.concat([averagespan, start, end, left_context_windows, right_context_windows], 2))

            wiqs_left5.append(
                tf.slice(wiq_pooled5, [0, 0, 0], tf.stack([-1, max_support_length - (window_size - 1), -1])))
            wiqs_left10.append(
                tf.slice(wiq_pooled10, [0, 0, 0], tf.stack([-1, max_support_length - (window_size - 1), -1])))
            wiqs_left20.append(
                tf.slice(wiq_pooled20, [0, 0, 0], tf.stack([-1, max_support_length - (window_size - 1), -1])))

            wiqs_right5.append(tf.slice(wiq_pooled5, [0, window_size + 5, 0], [-1, -1, -1]))
            wiqs_right10.append(tf.slice(wiq_pooled10, [0, window_size + 10, 0], [-1, -1, -1]))
            wiqs_right20.append(tf.slice(wiq_pooled20, [0, window_size + 20, 0], [-1, -1, -1]))

            spans.append(tf.stack([tf.range(0, max_support_length - (window_size - 1)),
                                   tf.range(window_size - 1, max_support_length)], 1))

        span_rep = tf.concat(span_rep, 1)
        span_rep.set_shape([None, None, input_size * 5])
        wiqs_left5 = tf.concat(wiqs_left5, 1)
        wiqs_left10 = tf.concat(wiqs_left10, 1)
        wiqs_left20 = tf.concat(wiqs_left20, 1)

        wiqs_right5 = tf.concat(wiqs_right5, 1)
        wiqs_right10 = tf.concat(wiqs_right10, 1)
        wiqs_right20 = tf.concat(wiqs_right20, 1)

        spans = tf.concat(spans, 0)

        # scoring
        with tf.variable_scope("question_rep"):
            question_rep = tf.layers.dense(question_rep, size, activation=tf.tanh)
        with tf.variable_scope("question_inter"):
            question_inter = tf.layers.dense(question_rep, size, activation=None)

        with tf.variable_scope("span_rep"):
            span_rep = tf.layers.dense(span_rep, size, activation=tf.tanh)

        span_question_rep = tf.concat([span_rep, tf.expand_dims(question_rep, 1) * span_rep,
                                       wiqs_left5, wiqs_left10, wiqs_left20,
                                       wiqs_right5, wiqs_right10, wiqs_right20], 2)
        span_question_rep.set_shape([None, None, 2 * size + 6 * 2])

        with tf.variable_scope("hidden"):
            h = tf.tanh(tf.layers.dense(span_question_rep, size, activation=None) + tf.expand_dims(question_inter, 1))

        with tf.variable_scope("scoring"):
            span_scores = tf.squeeze(tf.layers.dense(h, 1, activation=None), 2)

        best_span = tf.arg_max(span_scores, 1)
        predicted_span = tf.gather(spans, best_span)

        return span_scores, tf.tile(tf.expand_dims(spans, 0), tf.stack([batch_size, 1, 1])), predicted_span
