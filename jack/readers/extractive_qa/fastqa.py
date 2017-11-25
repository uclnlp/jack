"""
This file contains FastQA specific modules and ports
"""

from jack.core import *
from jack.readers.extractive_qa.answer_layer import fastqa_answer_layer
from jack.readers.extractive_qa.shared import XQAPorts, AbstractXQAModelModule
from jack.tfutil import misc
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.highway import highway_network


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
                      word_chars, word_char_length,
                      question_words, support_words,
                      word_in_question,
                      correct_start, answer2support, keep_prob, is_eval):
        with tf.variable_scope("fast_qa", initializer=tf.contrib.layers.xavier_initializer()):
            # Some helpers
            batch_size = tf.shape(question_length)[0]
            max_question_length = tf.reduce_max(question_length)
            support_mask = misc.mask_for_lengths(support_length)

            input_size = shared_resources.config["repr_dim_input"]
            size = shared_resources.config["repr_dim"]
            with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)

            # set shapes for inputs
            emb_question.set_shape([None, None, input_size])
            emb_support.set_shape([None, None, input_size])

            if with_char_embeddings:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    len(shared_resources.char_vocab), size, word_chars, word_char_length,
                    [question_words, support_words])

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

            wiq_w = tf.matmul(tf.gather(emb_question * v_wiqw, support2question), emb_support, adjoint_b=True)
            wiq_w = wiq_w + tf.expand_dims(support_mask, 1)

            question_binary_mask = tf.gather(tf.sequence_mask(question_length, dtype=tf.float32), support2question)
            wiq_w = tf.reduce_sum(tf.nn.softmax(wiq_w) * tf.expand_dims(question_binary_mask, 2), [1])

            # [B, L , 2]
            support_features = tf.concat([tf.expand_dims(word_in_question, 2), tf.expand_dims(wiq_w, 2)], 2)

            # highway layer to allow for interaction between concatenated embeddings
            if with_char_embeddings:
                with tf.variable_scope("char_embeddings") as vs:
                    emb_question = tf.layers.dense(emb_question, size, name="embeddings_projection")
                    emb_question = highway_network(emb_question, 1)
                    vs.reuse_variables()
                    emb_support = tf.layers.dense(emb_support, size, name="embeddings_projection")
                    emb_support = highway_network(emb_support, 1)

            emb_question, emb_support = tf.cond(is_eval,
                                                lambda: (emb_question, emb_support),
                                                lambda: (tf.nn.dropout(emb_question, keep_prob),
                                                         tf.nn.dropout(emb_support, keep_prob)))

            # extend embeddings with features
            emb_question_ext = tf.concat([emb_question, question_features], 2)
            emb_support_ext = tf.concat([emb_support, support_features], 2)

            # encode question and support
            encoder = shared_resources.config.get('encoder', 'lstm').lower()
            if encoder in ['lstm', 'sru', 'gru']:
                size = size + 2 if encoder == 'sru' else size  # to allow for use of residual in SRU
                encoded_question = self.rnn_encoder(size, emb_question_ext, question_length, encoder)
                encoded_support = self.rnn_encoder(size, emb_support_ext, support_length, encoder, reuse=True)
                projection_initializer = tf.constant_initializer(np.concatenate([np.eye(size), np.eye(size)]))
                encoded_question = tf.layers.dense(encoded_question, size,
                                                   kernel_initializer=projection_initializer, name='projection_q')
                encoded_support = tf.layers.dense(encoded_support, size,
                                                  kernel_initializer=projection_initializer, name='projection_s')
            else:
                raise ValueError("Only rnn ('lstm', 'sru', 'gru') encoder allowed for FastQA!")

            start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                fastqa_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                                    correct_start, support2question, answer2support, is_eval,
                                    beam_size=shared_resources.config.get("beam_size", 1),
                                    max_span_size=shared_resources.config.get("max_span_size", 16))

            span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

            return start_scores, end_scores, span
