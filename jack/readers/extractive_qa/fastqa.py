"""
This file contains FastQA specific modules and ports
"""

from jack.core import *
from jack.readers.extractive_qa.answer_layer import mlp_answer_layer
from jack.readers.extractive_qa.shared import XQAPorts, AbstractXQAModelModule
from jack.tfutil import misc
from jack.tfutil.dropout import fixed_dropout
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.highway import highway_network
from jack.tfutil.rnn import birnn_with_projection


class FastQAModule(AbstractXQAModelModule):
    _input_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                    XQAPorts.emb_support, XQAPorts.support_length, XQAPorts.support2question,
                    # char embedding inputs
                    XQAPorts.word_chars, XQAPorts.word_length,
                    XQAPorts.question_words, XQAPorts.support_words,
                    # feature input
                    XQAPorts.word_in_question,
                    # optional input, provided only during training
                    XQAPorts.correct_start_training, XQAPorts.answer2support_training,
                    XQAPorts.keep_prob, XQAPorts.is_eval]

    @property
    def input_ports(self):
        return self._input_ports

    def create_output(self, shared_resources, emb_question, question_length,
                      emb_support, support_length, support2question,
                      unique_word_chars, unique_word_char_length,
                      question_words2unique, support_words2unique,
                      word_in_question,
                      correct_start, answer2support, keep_prob, is_eval):
        """FastQA model.
        Args:
            shared_resources: has at least a field config (dict) with keys "rep_dim", "rep_dim_input"
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
            answer2support: [A], only during training, i.e., is_eval=False
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

            input_size = shared_resources.config["repr_dim_input"]
            size = shared_resources.config["repr_dim"]
            with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)

            # set shapes for inputs
            emb_question.set_shape([None, None, input_size])
            emb_support.set_shape([None, None, input_size])

            if with_char_embeddings:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    len(shared_resources.char_vocab), size, unique_word_chars, unique_word_char_length,
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

            start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                mlp_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                                 correct_start, support2question, answer2support, is_eval,
                                 beam_size=shared_resources.config.get("beam_size", 1))

            span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

            return start_scores, end_scores, span
