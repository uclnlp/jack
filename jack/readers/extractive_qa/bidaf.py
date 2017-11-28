import logging

import numpy as np
import tensorflow as tf

from jack.readers.extractive_qa.answer_layer import compute_spans
from jack.readers.extractive_qa.shared import AbstractXQAModelModule
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.highway import highway_network
from jack.tfutil.misc import mask_for_lengths

logger = logging.getLogger(__name__)


class BiDAF(AbstractXQAModelModule):
    def create_output(self, shared_resources, emb_question, question_length,
                      emb_support, support_length, support2question,
                      word_chars, word_char_length,
                      question_words, support_words,
                      answer2support, is_eval):
        with tf.variable_scope("bidaf", initializer=tf.contrib.layers.xavier_initializer()):
            # Some helpers
            max_support_length = tf.reduce_max(support_length)

            beam_size = 1
            beam_size = tf.cond(is_eval, lambda: tf.constant(beam_size, tf.int32), lambda: tf.constant(1, tf.int32))

            input_size = shared_resources.config["repr_dim_input"]
            size = shared_resources.config["repr_dim"]
            with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)

            # 1. char embeddings + word embeddings
            # set shapes for inputs
            emb_question.set_shape([None, None, input_size])
            emb_support.set_shape([None, None, input_size])

            # 1. + 2a. + 2b. 2a. char embeddings + conv + max pooling
            if with_char_embeddings:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(len(shared_resources.char_vocab),
                                                                            size,
                                                                            word_chars, word_char_length,
                                                                            [question_words,
                                                                             support_words])
                # 3. cat
                emb_question = tf.concat([emb_question, char_emb_question], 2)
                emb_support = tf.concat([emb_support, char_emb_support], 2)

                # highway layer to allow for interaction between concatenated embeddings
                # 3. highway
                # following bidaf notation here  (qq=question, xx=support)
                emb_question = highway_network(emb_question, 2, scope='question_highway')
                emb_support = highway_network(emb_support, 2, scope='support_highway')

            keep_prob = 1.0 - shared_resources.config.get("dropout", 0.0)
            emb_question, emb_support = tf.cond(
                is_eval,
                lambda: (emb_question, emb_support),
                lambda: (tf.nn.dropout(emb_question, keep_prob, noise_shape=[1, 1, emb_question.get_shape()[-1].value]),
                         tf.nn.dropout(emb_support, keep_prob, noise_shape=[1, 1, emb_question.get_shape()[-1].value]))
            )

            # 4. Context encoder
            # encode question and support
            encoder = shared_resources.config.get('encoder', 'lstm').lower()
            if encoder in ['lstm', 'sru', 'gru']:
                encoded_question = self.rnn_encoder(size, emb_question, question_length, encoder)
                encoded_support = self.rnn_encoder(size, emb_support, support_length, encoder, reuse=True)
                projection_initializer = tf.constant_initializer(np.concatenate([np.eye(size), np.eye(size)]))
                question = tf.layers.dense(encoded_question, size,
                                           kernel_initializer=projection_initializer, name='projection_q')
                support = tf.layers.dense(encoded_support, size,
                                          kernel_initializer=projection_initializer, name='projection_s')
            else:
                # follows https://openreview.net/pdf?id=HJRV1ZZAW
                question = self.conv_encoder(size, emb_question, num_layers=5,
                                             encoder_type='convnet', name='question_encoder')
                support = self.conv_encoder(size, emb_support, encoder_type='convnet',
                                            num_layers=5, name='support_encoder')

            # align with support
            question = tf.gather(question, support2question)

            # 5. a_i,j= w1 * s_i + w2 * q_j + w3 * q_j * s_i
            w_3 = tf.get_variable("w3", [size, 1])
            question_mul = question * tf.reshape(w_3, [1, 1, size])

            # S = [batch, support_length, question_length]
            S = (tf.einsum('ijk,ilk->ijl', support, question_mul) +
                 tf.layers.dense(support, 1) + tf.reshape(tf.layers.dense(question, 1), [-1, 1, tf.shape(question)[1]]))

            # support to question attention
            att_question = tf.nn.softmax(S, 2)  # softmax over question for each support token
            question_weighted = tf.einsum('ijl,ilk->ijk', att_question, question)

            # support to question attention
            # 1. filter important context words with max
            # 2. softmax over question to get the question words which are most relevant for the most relevant context words
            # max(S) = [batch, length1, length2] -> [ batch, length1] = most important context
            max_support = tf.reduce_max(S, 2)
            # softmax over question -> [batch, length1]
            support_attention = tf.nn.softmax(max_support, 1)
            # support attention * support = weighted support
            # [batch, length1] * [batch, length1, length2, 2*embedding] = [batch, 2*embedding]
            support_weighted = tf.einsum('ij,ijk->ik', support_attention, support)
            # tile to have the same dimension
            # [batch, 2*embedding] -> [batch, length2, 2*embedding]
            support_weighted = tf.expand_dims(support_weighted, 1)
            support_weighted = tf.tile(support_weighted, [1, max_support_length, 1])

            # 6 generate feature matrix
            # G(support, weighted question, weighted support)  = G(h, *u, *h) = [h, *u, mul(h, *u), mul(h, h*)] = [batch, length2, embedding*8]
            G = tf.concat([support, question_weighted, support * question_weighted, support * support_weighted], 2)

            # 7. BiLSTM(G) = M
            # interaction_encoded = M
            if encoder in ['lstm', 'sru', 'gru']:
                interaction_encoded = self.rnn_encoder(size, G, support_length, encoder, name='interaction_encoder')
            else:
                # follows https://openreview.net/pdf?id=HJRV1ZZAW
                interaction_encoded = self.conv_encoder(size, G, dilations=[1, 2, 4, 8, 16, 1, 1, 1],
                                                        encoder_type='gldr', name='interaction_encoder')
            start_encoded = tf.concat([interaction_encoded, G], 2)

            # BiLSTM(M) = M^2 = end_encoded
            if encoder in ['lstm', 'sru', 'gru']:
                end_encoded = self.rnn_encoder(size, start_encoded, support_length, encoder, name='end_encoder')
            else:
                # follows https://openreview.net/pdf?id=HJRV1ZZAW
                end_encoded = self.conv_encoder(size, start_encoded, num_layers=3,
                                                encoder_type='convnet', name='end_encoder')
            end_encoded = tf.concat([end_encoded, G], 2)
            # 8. logits
            start_scores = tf.squeeze(tf.layers.dense(start_encoded, 1, use_bias=False), 2)
            end_scores = tf.squeeze(tf.layers.dense(end_encoded, 1, use_bias=False), 2)

            # mask out-of-bounds slots by adding -1000
            support_mask = mask_for_lengths(support_length)
            start_scores = start_scores + support_mask
            end_scores = end_scores + support_mask

            start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                compute_spans(start_scores, end_scores, answer2support, is_eval, support2question,
                              beam_size=shared_resources.config.get("beam_size", 1),
                              max_span_size=shared_resources.config.get("max_span_size", 16))

            span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

            return start_scores, end_scores, span
