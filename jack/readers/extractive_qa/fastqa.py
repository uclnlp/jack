"""
This file contains FastQA specific modules and ports
"""

from typing import NamedTuple

from jack.core import *
from jack.readers.extractive_qa.shared import XQAPorts, AbstractXQAModelModule
from jack.tfutil import misc
from jack.tfutil.dropout import fixed_dropout
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.highway import highway_network
from jack.tfutil.rnn import birnn_with_projection

FastQAAnnotation = NamedTuple('FastQAAnnotation', [
    ('question_tokens', List[str]),
    ('question_ids', List[int]),
    ('question_length', int),
    ('question_embeddings', np.ndarray),
    ('support_tokens', List[str]),
    ('support_ids', List[int]),
    ('support_length', int),
    ('support_embeddings', np.ndarray),
    ('word_in_question', List[float]),
    ('token_offsets', List[int]),
    ('answer_spans', Optional[List[Tuple[int, int]]]),
])


class FastQAModule(AbstractXQAModelModule):
    _input_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                    XQAPorts.emb_support, XQAPorts.support_length,
                    # char embedding inputs
                    XQAPorts.unique_word_chars, XQAPorts.unique_word_char_length,
                    XQAPorts.question_words2unique, XQAPorts.support_words2unique,
                    # feature input
                    XQAPorts.word_in_question,
                    # optional input, provided only during training
                    XQAPorts.correct_start_training, XQAPorts.answer2question_training,
                    XQAPorts.keep_prob, XQAPorts.is_eval]

    @property
    def input_ports(self):
        return self._input_ports

    def create_output(self, shared_vocab_config, emb_question, question_length,
                      emb_support, support_length,
                      unique_word_chars, unique_word_char_length,
                      question_words2unique, support_words2unique,
                      word_in_question,
                      correct_start, answer2question, keep_prob, is_eval):
        """FastQA model.
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
            support_mask = misc.mask_for_lengths(support_length)
            question_binary_mask = misc.mask_for_lengths(question_length, mask_right=False, value=1.0)

            input_size = shared_vocab_config.config["repr_dim_input"]
            size = shared_vocab_config.config["repr_dim"]
            with_char_embeddings = shared_vocab_config.config.get("with_char_embeddings", False)

            # set shapes for inputs
            emb_question.set_shape([None, None, input_size])
            emb_support.set_shape([None, None, input_size])

            if with_char_embeddings:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    shared_vocab_config.char_vocab, size, unique_word_chars, unique_word_char_length,
                    [question_words2unique, support_words2unique])

                emb_question = tf.concat([emb_question, char_emb_question], 2)
                emb_support = tf.concat([emb_support, char_emb_support], 2)
                input_size += size

                # set shapes for inputs
                emb_question.set_shape([None, None, input_size])
                emb_support.set_shape([None, None, input_size])

            # compute encoder features
            question_features = tf.ones(tf.stack([batch_size, max_question_length, 2]))

            v_wiqw = tf.get_variable("v_wiq_w", [1, 1, input_size],
                                     initializer=tf.constant_initializer(1.0))

            wiq_w = tf.matmul(emb_question * v_wiqw, emb_support, adjoint_b=True)
            wiq_w = wiq_w + tf.expand_dims(support_mask, 1)

            wiq_w = tf.reduce_sum(tf.nn.softmax(wiq_w) * tf.expand_dims(question_binary_mask, 2), [1])

            # [B, L , 2]
            support_features = tf.concat([tf.expand_dims(word_in_question, 2), tf.expand_dims(wiq_w, 2)], 2)

            # highway layer to allow for interaction between concatenated embeddings
            if with_char_embeddings:
                all_embedded = tf.concat([emb_question, emb_support], 1)
                all_embedded = tf.contrib.layers.fully_connected(all_embedded, size,
                                                                 activation_fn=None,
                                                                 weights_initializer=None,
                                                                 biases_initializer=None,
                                                                 scope="embeddings_projection")

                all_embedded_hw = highway_network(all_embedded, 1)

                emb_question = tf.slice(all_embedded_hw, [0, 0, 0], tf.stack([-1, max_question_length, -1]))
                emb_support = tf.slice(all_embedded_hw, tf.stack([0, max_question_length, 0]), [-1, -1, -1])

                emb_question.set_shape([None, None, size])
                emb_support.set_shape([None, None, size])

            # variational dropout
            dropout_shape = tf.unstack(tf.shape(emb_question))
            dropout_shape[1] = 1

            [emb_question, emb_support] = tf.cond(is_eval,
                                                  lambda: [emb_question, emb_support],
                                                  lambda: fixed_dropout([emb_question, emb_support],
                                                                        keep_prob, dropout_shape))

            # extend embeddings with features
            emb_question_ext = tf.concat([emb_question, question_features], 2)
            emb_support_ext = tf.concat([emb_support, support_features], 2)

            # encode question and support
            rnn = tf.contrib.rnn.LSTMBlockFusedCell
            encoded_question = birnn_with_projection(size, rnn, emb_question_ext, question_length,
                                                     projection_scope="question_proj")

            encoded_support = birnn_with_projection(size, rnn, emb_support_ext, support_length,
                                                    share_rnn=True, projection_scope="support_proj")

            start_scores, end_scores, predicted_start_pointer, predicted_end_pointer = \
                fastqa_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                                    correct_start, answer2question, is_eval,
                                    beam_size=shared_vocab_config.config.get("beam_size", 1))

            span = tf.stack([predicted_start_pointer, predicted_end_pointer], 1)

            return start_scores, end_scores, span


# ANSWER LAYER
def fastqa_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                        correct_start, answer2question, is_eval, beam_size=1):
    beam_size = tf.cond(is_eval, lambda: tf.constant(beam_size, tf.int32), lambda: tf.constant(1, tf.int32))
    batch_size = tf.shape(question_length)[0]
    answer2question = tf.cond(is_eval, lambda: tf.range(0, batch_size, dtype=tf.int32), lambda: answer2question)
    input_size = encoded_support.get_shape()[-1].value
    support_states_flat = tf.reshape(encoded_support, [-1, input_size])

    # computing single time attention over question
    attention_scores = tf.contrib.layers.fully_connected(encoded_question, 1,
                                                         activation_fn=None,
                                                         weights_initializer=None,
                                                         biases_initializer=None,
                                                         scope="question_attention")
    q_mask = misc.mask_for_lengths(question_length)
    attention_scores = attention_scores + tf.expand_dims(q_mask, 2)
    question_attention_weights = tf.nn.softmax(attention_scores, 1, name="question_attention_weights")
    question_state = tf.reduce_sum(question_attention_weights * encoded_question, [1])

    # Prediction
    # start
    start_input = tf.concat([tf.expand_dims(question_state, 1) * encoded_support,
                             encoded_support], 2)

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

    support_mask = misc.mask_for_lengths(support_length)
    start_scores = start_scores + support_mask

    # probs are needed during beam search
    start_probs = tf.nn.softmax(start_scores)

    predicted_start_probs, predicted_start_pointer = tf.nn.top_k(start_probs, beam_size)

    # use correct start during training, because p(end|start) should be optimized
    predicted_start_pointer = tf.gather(predicted_start_pointer, answer2question)
    predicted_start_probs = tf.gather(predicted_start_probs, answer2question)

    start_pointer = tf.cond(is_eval, lambda: predicted_start_pointer, lambda: tf.expand_dims(correct_start, 1))

    # flatten again
    start_pointer = tf.reshape(start_pointer, [-1])
    answer2questionwithbeam = tf.reshape(tf.tile(tf.expand_dims(answer2question, 1), tf.stack([1, beam_size])), [-1])

    offsets = tf.cast(tf.range(0, batch_size) * tf.reduce_max(support_length), dtype=tf.int32)
    offsets = tf.gather(offsets, answer2questionwithbeam)
    u_s = tf.gather(support_states_flat, start_pointer + offsets)

    start_scores = tf.gather(start_scores, answer2questionwithbeam)
    start_input = tf.gather(start_input, answer2questionwithbeam)
    encoded_support = tf.gather(encoded_support, answer2questionwithbeam)
    question_state = tf.gather(question_state, answer2questionwithbeam)
    support_mask = tf.gather(support_mask, answer2questionwithbeam)

    # end
    end_input = tf.concat([tf.expand_dims(u_s, 1) * encoded_support, start_input], 2)

    q_end_inter = tf.contrib.layers.fully_connected(tf.concat([question_state, u_s], 1), size,
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

    def mask_with_start(scores):
        return scores + misc.mask_for_lengths(tf.cast(start_pointer, tf.int32),
                                              tf.reduce_max(support_length),
                                              mask_right=False)

    end_scores = tf.cond(is_eval, lambda: mask_with_start(end_scores), lambda: end_scores)

    # probs are needed during beam search
    end_probs = tf.nn.softmax(end_scores)
    predicted_end_probs, predicted_end_pointer = tf.nn.top_k(end_probs, 1)
    predicted_end_probs = tf.reshape(predicted_end_probs, tf.stack([-1, beam_size]))
    predicted_end_pointer = tf.reshape(predicted_end_pointer, tf.stack([-1, beam_size]))

    predicted_idx = tf.cast(tf.argmax(predicted_start_probs * predicted_end_probs, 1), tf.int32)
    predicted_idx = tf.stack([tf.range(0, tf.shape(answer2question)[0], dtype=tf.int32), predicted_idx], 1)

    predicted_start_pointer = tf.gather_nd(predicted_start_pointer, predicted_idx)
    predicted_end_pointer = tf.gather_nd(predicted_end_pointer, predicted_idx)

    return start_scores, end_scores, predicted_start_pointer, predicted_end_pointer
