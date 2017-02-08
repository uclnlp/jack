"""
This file contains FastQA specific modules and ports
"""

import random
import re

from jtr.jack import *
from jtr.jack.fun import simple_model_module, no_shared_resources
from jtr.jack.tasks.xqa.shared import XQAPorts
from jtr.jack.tasks.xqa.util import token_to_char_offsets
from jtr.jack.tf_fun.dropout import fixed_dropout
from jtr.jack.tf_fun.embedding import conv_char_embedding_alt
from jtr.jack.tf_fun.highway import highway_network
from jtr.jack.tf_fun.rnn import birnn_with_projection
from jtr.jack.tf_fun.xqa import xqa_min_crossentropy_loss
from jtr.preprocess.batch import GeneratorWithRestart
from jtr.preprocess.map import deep_map, numpify
from jtr.util import tfutil


class FastQAPorts:
    """
    It is good practice define all ports needed for a single model jointly, to get an overview
    """

    # We feed embeddings directly
    emb_question = FlatPorts.Misc.embedded_question
    question_length = FlatPorts.Input.question_length
    emb_support = FlatPorts.Misc.embedded_support
    support_length = FlatPorts.Input.support_length

    # but also ids, for char-based embeddings
    unique_word_chars = TensorPort(tf.int32, [None, None], "question_chars",
                                   "Represents questions using symbol vectors",
                                   "[U, max_num_chars]")
    unique_word_char_length = TensorPort(tf.int32, [None], "question_char_length",
                                         "Represents questions using symbol vectors",
                                         "[U]")
    question_words2unique = TensorPort(tf.int32, [None, None], "question_words2unique",
                                       "Represents support using symbol vectors",
                                       "[batch_size, max_num_question_tokens]")
    support_words2unique = TensorPort(tf.int32, [None, None], "support_words2unique",
                                      "Represents support using symbol vectors",
                                      "[batch_size, max_num_support_tokens, max]")

    keep_prob = Ports.Input.keep_prob
    is_eval = Ports.Input.is_eval

    # This feature is model specific and thus, not part of the conventional Ports
    word_in_question = TensorPort(tf.float32, [None, None], "word_in_question_feature",
                                  "Represents a 1/0 feature for all context tokens denoting"
                                  " whether it is part of the question or not",
                                  "[Q, support_length]")

    correct_start_training = TensorPortWithDefault(np.array([0], np.int64), tf.int64, [None], "correct_start_training",
                                                   "Represents the correct start of the span which is given to the"
                                                   "model during training for use to predicting end.",
                                                   "[A]")

    answer2question_training = TensorPortWithDefault([0], tf.int32, [None], "answer2question_training",
                                                     "Represents mapping to question idx per answer, which is used "
                                                     "together with correct_start_training during training.",
                                                     "[A]")

    # output ports
    start_scores = FlatPorts.Prediction.start_scores
    end_scores = FlatPorts.Prediction.end_scores
    span_prediction = FlatPorts.Prediction.answer_span
    token_char_offsets = XQAPorts.token_char_offsets

    # ports used during training
    answer2question = FlatPorts.Input.answer2question
    answer_span = FlatPorts.Target.answer_span


class FastQAInputModule(InputModule):
    def __init__(self, shared_vocab_config):
        assert isinstance(shared_vocab_config, SharedVocabAndConfig), \
            "shared_resources for FastQAInputModule must be an instance of SharedVocabAndConfig"
        self.shared_vocab_config = shared_vocab_config

    __pattern = re.compile('\w+|[^\w\s]')

    @staticmethod
    def tokenize(text):
        return FastQAInputModule.__pattern.findall(text)

    def _get_emb(self, idx):
        if idx < self.emb_matrix.shape[0]:
            return self.emb_matrix[idx]
        else:
            return self.default_vec

    @property
    def output_ports(self) -> List[TensorPort]:
        return [FastQAPorts.emb_question, FastQAPorts.question_length,
                FastQAPorts.emb_support, FastQAPorts.support_length,
                # char
                FastQAPorts.unique_word_chars, FastQAPorts.unique_word_char_length,
                FastQAPorts.question_words2unique, FastQAPorts.support_words2unique,
                # features
                FastQAPorts.word_in_question,
                # optional, only during training
                FastQAPorts.correct_start_training, FastQAPorts.answer2question_training,
                FastQAPorts.keep_prob, FastQAPorts.is_eval,
                # for output module
                FastQAPorts.token_char_offsets]

    @property
    def training_ports(self) -> List[TensorPort]:
        return [FastQAPorts.answer_span, FastQAPorts.answer2question]

    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        # create character vocab + word lengths + char ids per word
        vocab = self.shared_vocab_config.vocab
        char_vocab = dict()
        char_vocab["PAD"] = 0
        for i in range(max(vocab.id2sym.keys()) + 1):
            w = vocab.id2sym.get(i)
            if w is not None:
                for c in w:
                    if c not in char_vocab:
                        char_vocab[c] = len(char_vocab)
        self.shared_vocab_config.config["char_vocab"] = char_vocab
        # Assumes that vocab and embeddings are given during creation
        self.setup()
        return self.shared_vocab_config

    def setup(self):
        vocab = self.shared_vocab_config.vocab
        config = self.shared_vocab_config.config
        self.batch_size = config.get("batch_size", 1)
        self.dropout = config.get("dropout", 1)
        self._rng = random.Random(config.get("seed", 123))
        self.emb_matrix = vocab.emb.lookup
        self.default_vec = np.zeros([vocab.emb_length])
        self.char_vocab = self.shared_vocab_config.config["char_vocab"]

    def prepare_data(self, dataset, with_answers=False):
        corpus = {"support": [], "question": []}
        for d in dataset:
            if isinstance(d, QASetting):
                qa_setting = d
            else:
                qa_setting, answer = d

            if self.shared_vocab_config.config.get("lowercase", False):
                corpus["support"].append(" ".join(qa_setting.support).lower())
                corpus["question"].append(qa_setting.question.lower())
            else:
                corpus["support"].append(" ".join(qa_setting.support))
                corpus["question"].append(qa_setting.question)

        corpus_tokenized = deep_map(corpus, self.tokenize, ['question', 'support'])
        corpus_ids = deep_map(corpus_tokenized, self.shared_vocab_config.vocab, ['question', 'support'])

        word_in_question = []
        question_lengths = []
        support_lengths = []
        token_offsets = []
        answer_spans = []

        for i, (q, s) in enumerate(zip(corpus_tokenized["question"], corpus_tokenized["support"])):
            support_lengths.append(len(s))
            question_lengths.append(len(q))

            # char to token offsets
            support = corpus["support"][i]
            offsets = token_to_char_offsets(support, s)
            token_offsets.append(offsets)

            # word in question feature
            wiq = []
            for token in s:
                wiq.append(float(token in q))
            word_in_question.append(wiq)

            if with_answers:
                answers = dataset[i][1]
                spans = []
                for a in answers:
                    start = 0
                    while start < len(offsets) and offsets[start] < a.span[0]:
                        start += 1

                    if start == len(offsets):
                        continue

                    end = start
                    while end + 1 < len(offsets) and offsets[end + 1] < a.span[1]:
                        end += 1
                    if (start, end) not in spans:
                        spans.append((start, end))
                answer_spans.append(spans)

        return corpus_tokenized["question"], corpus_ids["question"], question_lengths, \
               corpus_tokenized["support"], corpus_ids["support"], support_lengths, \
               word_in_question, token_offsets, answer_spans

    def unique_words(self, q_tokenized, s_tokenized, indices=None):
        indices = indices or range(len(q_tokenized))

        unique_words_set = dict()
        unique_words = list()
        unique_word_lengths = list()
        question2unique = list()
        support2unique = list()

        for j in indices:
            q2u = list()
            for w in q_tokenized[j]:
                if w not in unique_words_set:
                    unique_word_lengths.append(len(w))
                    unique_words.append([self.char_vocab.get(c, 0) for c in w])
                    unique_words_set[w] = len(unique_words_set)
                q2u.append(unique_words_set[w])
            question2unique.append(q2u)
            s2u = list()
            for w in s_tokenized[j]:
                if w not in unique_words_set:
                    unique_word_lengths.append(len(w))
                    unique_words.append([self.char_vocab.get(c, 0) for c in w])
                    unique_words_set[w] = len(unique_words_set)
                s2u.append(unique_words_set[w])
            support2unique.append(s2u)

        return unique_words, unique_word_lengths, question2unique, support2unique

    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]], is_eval: bool) \
            -> Iterable[Mapping[TensorPort, np.ndarray]]:
        q_tokenized, q_ids, q_lengths, s_tokenized, s_ids, s_lengths, \
        word_in_question, token_offsets, answer_spans = self.prepare_data(dataset, with_answers=True)

        emb_supports = np.zeros([self.batch_size, max(s_lengths), self.emb_matrix.shape[1]])
        emb_questions = np.zeros([self.batch_size, max(q_lengths), self.emb_matrix.shape[1]])

        def batch_generator():
            todo = list(range(len(q_ids)))
            self._rng.shuffle(todo)
            while todo:
                support_lengths = list()
                question_lengths = list()
                wiq = list()
                spans = list()
                span2question = []
                offsets = []

                unique_words, unique_word_lengths, question2unique, support2unique = \
                    self.unique_words(q_tokenized, s_tokenized, todo[:self.batch_size])

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
                    spans.extend(answer_spans[j])
                    span2question.extend(i for _ in answer_spans[j])
                    wiq.append(word_in_question[j])
                    offsets.append(token_offsets[j])

                batch_size = len(question_lengths)
                output = {
                    FastQAPorts.unique_word_chars: unique_words,
                    FastQAPorts.unique_word_char_length: unique_word_lengths,
                    FastQAPorts.question_words2unique: question2unique,
                    FastQAPorts.support_words2unique: support2unique,
                    FastQAPorts.emb_support: emb_supports[:batch_size, :max(support_lengths), :],
                    FastQAPorts.support_length: support_lengths,
                    FastQAPorts.emb_question: emb_questions[:batch_size, :max(question_lengths), :],
                    FastQAPorts.question_length: question_lengths,
                    FastQAPorts.word_in_question: wiq,
                    FastQAPorts.answer_span: spans,
                    FastQAPorts.correct_start_training: [] if is_eval else [s[0] for s in spans],
                    FastQAPorts.answer2question: span2question,
                    FastQAPorts.answer2question_training: [] if is_eval else span2question,
                    FastQAPorts.keep_prob: 1.0 if is_eval else 1 - self.dropout,
                    FastQAPorts.is_eval: is_eval,
                    FastQAPorts.token_char_offsets: offsets
                }

                # we can only numpify in here, because bucketing is not possible prior
                batch = numpify(output, keys=[FastQAPorts.unique_word_chars,
                                              FastQAPorts.question_words2unique, FastQAPorts.support_words2unique,
                                              FastQAPorts.word_in_question, FastQAPorts.token_char_offsets])
                todo = todo[self.batch_size:]
                yield batch

        return GeneratorWithRestart(batch_generator)

    def __call__(self, qa_settings: List[QASetting]) -> Mapping[TensorPort, np.ndarray]:
        q_tokenized, q_ids, q_lengths, s_tokenized, s_ids, s_lengths, \
        word_in_question, token_offsets, answer_spans = self.prepare_data(qa_settings, with_answers=False)

        unique_words, unique_word_lengths, question2unique, support2unique = self.unique_words(q_tokenized, s_tokenized)

        batch_size = len(qa_settings)
        emb_supports = np.zeros([batch_size, max(s_lengths), self.emb_matrix.shape[1]])
        emb_questions = np.zeros([batch_size, max(q_lengths), self.emb_matrix.shape[1]])

        for i, q in enumerate(q_ids):
            for k, v in enumerate(s_ids[i]):
                emb_supports[i, k] = self._get_emb(v)
            for k, v in enumerate(q):
                emb_questions[i, k] = self._get_emb(v)

        output = {
            FastQAPorts.unique_word_chars: unique_words,
            FastQAPorts.unique_word_char_length: unique_word_lengths,
            FastQAPorts.question_words2unique: question2unique,
            FastQAPorts.support_words2unique: support2unique,
            FastQAPorts.emb_support: emb_supports,
            FastQAPorts.support_length: s_lengths,
            FastQAPorts.emb_question: emb_questions,
            FastQAPorts.question_length: q_lengths,
            FastQAPorts.word_in_question: word_in_question,
            FastQAPorts.token_char_offsets: token_offsets
        }

        output = numpify(output, keys=[FastQAPorts.unique_word_chars, FastQAPorts.question_words2unique,
                                       FastQAPorts.support_words2unique, FastQAPorts.word_in_question,
                                       FastQAPorts.token_char_offsets])

        return output  # FastQA model module factory method, like fastqa.model.fastqa_model


fastqa_like_model_module_factory = simple_model_module(
    input_ports=[FastQAPorts.emb_question, FastQAPorts.question_length,
                 FastQAPorts.emb_support, FastQAPorts.support_length,
                 # char embedding inputs
                 FastQAPorts.unique_word_chars, FastQAPorts.unique_word_char_length,
                 FastQAPorts.question_words2unique, FastQAPorts.support_words2unique,
                 # feature input
                 FastQAPorts.word_in_question,
                 # optional input, provided only during training
                 FastQAPorts.correct_start_training, FastQAPorts.answer2question_training,
                 FastQAPorts.keep_prob, FastQAPorts.is_eval],
    output_ports=[FastQAPorts.start_scores, FastQAPorts.end_scores,
                  FastQAPorts.span_prediction],
    training_input_ports=[FastQAPorts.start_scores, FastQAPorts.end_scores,
                          FastQAPorts.answer_span, FastQAPorts.answer2question],
    training_output_ports=[Ports.loss])


def fastqa_like_with_min_crossentropy_loss_factory(shared_resources, f):
    return fastqa_like_model_module_factory(shared_resources, f, no_shared_resources(xqa_min_crossentropy_loss))


# Very specialized and therefore not sharable  TF code for fast qa model.
def fatqa_model_module(shared_vocab_config):
    return fastqa_like_with_min_crossentropy_loss_factory(shared_vocab_config, fastqa_model)


def fastqa_model(shared_vocab_config, emb_question, question_length,
                 emb_support, support_length,
                 unique_word_chars, unique_word_char_length,
                 question_words2unique, support_words2unique,
                 word_in_question,
                 correct_start, answer2question, keep_prob, is_eval):
    """
    fast_qa model
    Args:
        shared_vocab_config: has at least a field config (dict) with keys "rep_dim", "rep_input_dim"
        emb_question: [Q, L_q, N]
        question_length: [Q]
        emb_support: [Q, L_s, N]
        support_length: [Q]
        unique_word_chars
        unique_word_char_length
        question_words2unique
        support_words2unique
        word_in_question: [Q, L_s]
        correct_start: [A], only during training, i.e., is_eval=False
        answer2question: [A], only during training, i.e., is_eval=False
        keep_prob: []
        is_eval: []

    Returns:
        start_scores [B, L_s, N], end_scores [B, L_s, N], span_prediction [B, 2]
    """
    with tf.variable_scope("fast_qa", initializer=tf.contrib.layers.xavier_initializer()):
        # Some helpers
        batch_size = tf.shape(question_length)[0]
        max_question_length = tf.reduce_max(question_length)
        support_mask = tfutil.mask_for_lengths(support_length, batch_size)
        question_binary_mask = tfutil.mask_for_lengths(question_length, batch_size, mask_right=False, value=1.0)

        input_size = shared_vocab_config.config["repr_input_dim"]
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

            emb_question = tf.concat(2, [emb_question, char_emb_question])
            emb_support = tf.concat(2, [emb_support, char_emb_support])
            input_size += size

            # set shapes for inputs
            emb_question.set_shape([None, None, input_size])
            emb_support.set_shape([None, None, input_size])

        # compute encoder features
        question_features = tf.ones(tf.pack([batch_size, max_question_length, 2]))

        v_wiqw = tf.get_variable("v_wiq_w", [1, 1, input_size],
                                 initializer=tf.constant_initializer(1.0))

        wiq_w = tf.batch_matmul(emb_question * v_wiqw, emb_support, adj_y=True)
        wiq_w = wiq_w + tf.expand_dims(support_mask, 1)

        wiq_w = tf.reduce_sum(tf.nn.softmax(wiq_w) * tf.expand_dims(question_binary_mask, 2), [1])

        # [B, L , 2]
        support_features = tf.concat(2, [tf.expand_dims(word_in_question, 2), tf.expand_dims(wiq_w, 2)])

        # highway layer to allow for interaction between concatenated embeddings
        if with_char_embeddings:
            all_embedded = tf.concat(1, [emb_question, emb_support])
            all_embedded = tf.contrib.layers.fully_connected(all_embedded, size,
                                                             activation_fn=None,
                                                             weights_initializer=None,
                                                             biases_initializer=None,
                                                             scope="embeddings_projection")

            all_embedded_hw = highway_network(all_embedded, 1)

            emb_question = tf.slice(all_embedded_hw, [0, 0, 0], tf.pack([-1, max_question_length, -1]))
            emb_support = tf.slice(all_embedded_hw, tf.pack([0, max_question_length, 0]), [-1, -1, -1])

            emb_question.set_shape([None, None, size])
            emb_support.set_shape([None, None, size])

        # variational dropout
        dropout_shape = tf.unpack(tf.shape(emb_question))
        dropout_shape[1] = 1

        [emb_question, emb_support] = tf.cond(is_eval,
                                              lambda: [emb_question, emb_support],
                                              lambda: fixed_dropout([emb_question, emb_support],
                                                                    keep_prob, dropout_shape))

        # extend embeddings with features
        emb_question_ext = tf.concat(2, [emb_question, question_features])
        emb_support_ext = tf.concat(2, [emb_support, support_features])

        # encode question and support
        rnn = tf.contrib.rnn.LSTMBlockFusedCell
        encoded_question = birnn_with_projection(size, rnn, emb_question_ext, question_length,
                                                 projection_scope="question_proj")

        encoded_support = birnn_with_projection(size, rnn, emb_support_ext, support_length,
                                                share_rnn=True, projection_scope="support_proj")

        start_scores, end_scores, predicted_start_pointer, predicted_end_pointer = \
            fastqa_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                                correct_start, answer2question, is_eval)

        span = tf.concat(1, [tf.expand_dims(predicted_start_pointer, 1),
                             tf.expand_dims(predicted_end_pointer, 1)])

        return start_scores, end_scores, span


# ANSWER LAYER
def fastqa_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                        correct_start, answer2question, is_eval):
    batch_size = tf.shape(question_length)[0]
    input_size = encoded_support.get_shape()[-1].value
    support_states_flat = tf.reshape(encoded_support, [-1, input_size])

    # computing single time attention over question
    attention_scores = tf.contrib.layers.fully_connected(encoded_question, 1,
                                                         activation_fn=None,
                                                         weights_initializer=None,
                                                         biases_initializer=None,
                                                         scope="question_attention")
    q_mask = tfutil.mask_for_lengths(question_length, batch_size)
    attention_scores = attention_scores + tf.expand_dims(q_mask, 2)
    question_attention_weights = tf.nn.softmax(attention_scores, 1, name="question_attention_weights")
    question_state = tf.reduce_sum(question_attention_weights * encoded_question, [1])

    # Prediction
    # start
    start_input = tf.concat(2, [tf.expand_dims(question_state, 1) * encoded_support,
                                encoded_support])

    q_start_inter = tf.contrib.layers.fully_connected(question_state, size,
                                                      activation_fn=None,
                                                      weights_initializer=None,
                                                      scope="q_start_inter")

    q_start_state = tf.contrib.layers.fully_connected(start_input, size,
                                                      activation_fn=None,
                                                      weights_initializer=None,
                                                      biases_initializer=None,
                                                      scope="q_start") + tf.expand_dims(q_start_inter, 1)

    start_scores = tf.contrib.layers.fully_connected(tf.nn.relu(q_start_state), 1,
                                                     activation_fn=None,
                                                     weights_initializer=None,
                                                     biases_initializer=None,
                                                     scope="start_scores")
    start_scores = tf.squeeze(start_scores, [2])

    support_mask = tfutil.mask_for_lengths(support_length, batch_size)
    start_scores = start_scores + support_mask

    predicted_start_pointer = tf.argmax(start_scores, 1)

    # gather states for training, where spans should be predicted using multiple correct start per answer
    def align_tensor_with_answers_per_question(t):
        return tf.cond(is_eval, lambda: t, lambda: tf.gather(t, answer2question))

    # use correct start during training, because p(end|start) should be optimized
    predicted_start_pointer = align_tensor_with_answers_per_question(predicted_start_pointer)
    start_pointer = tf.cond(is_eval, lambda: predicted_start_pointer, lambda: correct_start)

    offsets = tf.cast(tf.range(0, batch_size) * tf.reduce_max(support_length), dtype=tf.int64)
    offsets = align_tensor_with_answers_per_question(offsets)
    u_s = tf.gather(support_states_flat, start_pointer + offsets)

    start_scores = align_tensor_with_answers_per_question(start_scores)
    start_input = align_tensor_with_answers_per_question(start_input)
    encoded_support = align_tensor_with_answers_per_question(encoded_support)
    question_state = align_tensor_with_answers_per_question(question_state)
    support_mask = align_tensor_with_answers_per_question(support_mask)

    # end
    end_input = tf.concat(2, [tf.expand_dims(u_s, 1) * encoded_support, start_input])

    q_end_inter = tf.contrib.layers.fully_connected(tf.concat(1, [question_state, u_s]), size,
                                                    activation_fn=None,
                                                    weights_initializer=None,
                                                    scope="q_end_inter")

    q_end_state = tf.contrib.layers.fully_connected(end_input, size,
                                                    activation_fn=None,
                                                    weights_initializer=None,
                                                    biases_initializer=None,
                                                    scope="q_end") + tf.expand_dims(q_end_inter, 1)

    end_scores = tf.contrib.layers.fully_connected(tf.nn.relu(q_end_state), 1,
                                                   activation_fn=None,
                                                   weights_initializer=None,
                                                   biases_initializer=None,
                                                   scope="end_scores")
    end_scores = tf.squeeze(end_scores, [2])
    end_scores = end_scores + support_mask
    end_scores = tf.cond(is_eval,
                         lambda: end_scores + tfutil.mask_for_lengths(tf.cast(predicted_start_pointer, tf.int32),
                                                                      batch_size, tf.reduce_max(support_length),
                                                                      mask_right=False),
                         lambda: end_scores)

    predicted_end_pointer = tf.argmax(end_scores, 1)

    return start_scores, end_scores, predicted_start_pointer, predicted_end_pointer
