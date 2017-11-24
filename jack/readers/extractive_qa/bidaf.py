import logging

import tensorflow as tf

from jack.readers.extractive_qa.answer_layer import compute_spans
from jack.readers.extractive_qa.shared import AbstractXQAModelModule
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.highway import highway_network
from jack.tfutil.misc import mask_for_lengths
from jack.tfutil.rnn import fused_birnn

logger = logging.getLogger(__name__)


class BiDAF(AbstractXQAModelModule):
    def create_output(self, shared_resources, emb_question, question_length,
                      emb_support, support_length, support2question,
                      word_chars, word_char_length,
                      question_words, support_words,
                      answer2support, keep_prob, is_eval):
        # 1. char embeddings + word embeddings
        # 2a. conv char embeddings
        # 2b. pool char embeddings
        # 3. cat + highway
        # 4. BiLSTM
        # 5. cat
        # 6. biattention
        # 6a. create matrix of question support attentions
        # 6b. generate feature matrix
        # 7. combine
        # 8. BiLSTM
        # 9. double cross-entropy loss
        with tf.variable_scope("bidaf", initializer=tf.contrib.layers.xavier_initializer()):
            # Some helpers
            max_question_length = tf.reduce_max(question_length)
            max_support_length = tf.reduce_max(support_length)

            beam_size = 1
            beam_size = tf.cond(is_eval, lambda: tf.constant(beam_size, tf.int32), lambda: tf.constant(1, tf.int32))

            input_size = shared_resources.config["repr_dim_input"]
            size = shared_resources.config["repr_dim"]
            with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
            W = tf.get_variable("biattention_weight", [size * 6])
            W_start = tf.get_variable("start_index_weight", [size * 10])
            W_end = tf.get_variable("end_index_weight", [size * 10])

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
                input_size += size

            # highway layer to allow for interaction between concatenated embeddings
            # 3. highway
            # following bidaf notation here  (qq=question, xx=support)
            highway_question = highway_network(emb_question, 2, scope='question_highway')
            highway_support = highway_network(emb_support, 2, scope='support_highway')

            # emb_question = tf.slice(highway_question, [0, 0, 0], tf.stack([-1, max_question_length, -1]))
            # emb_support = tf.slice(all_embedded_hw, tf.stack([0, max_question_length, 0]), [-1, -1, -1])

            # emb_question.set_shape([None, None, size])
            # emb_support.set_shape([None, None, size])


            # 4. BiLSTM
            cell1 = tf.contrib.rnn.LSTMBlockFusedCell(size)
            encoded_question = fused_birnn(cell1, highway_question, question_length, dtype=tf.float32, time_major=False,
                                           scope='question_encoding')[0]
            question = tf.concat(encoded_question, 2)

            cell2 = tf.contrib.rnn.LSTMBlockFusedCell(size)
            encoded_support = fused_birnn(cell2, highway_support, support_length, dtype=tf.float32, time_major=False,
                                          scope='support_encoding')[0]
            support = tf.concat(encoded_support, 2)

            # align with support
            question = tf.gather(question, support2question)

            # 5. a_i,j= w1 * s_i + w2 * q_j + w3 * q_j * s_i
            w_3 = tf.get_variable("w3", [2 * size, 1])
            question_mul = question * tf.reshape(w_3, [1, 1, 2 * size])

            S = (tf.einsum('ijk,ilk->ijl', support, question_mul) +
                 tf.layers.dense(support, 1) + tf.reshape(tf.layers.dense(question, 1), [-1, 1, tf.shape(question)[1]]))

            # S = [batch, length1, length2]
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
            # start_emb = M
            cell3 = tf.contrib.rnn.LSTMBlockFusedCell(size)
            start_emb = \
                fused_birnn(cell3, G, support_length, dtype=tf.float32, time_major=False, scope='start_emb')[0]
            start_emb = tf.concat(start_emb, 2)
            start_emb = tf.concat([start_emb, G], 2)
            # BiLSTM(M) = M^2 = end_emb
            cell4 = tf.contrib.rnn.LSTMBlockFusedCell(size)
            end_emb = \
                fused_birnn(cell4, start_emb, support_length, dtype=tf.float32, time_major=False, scope='end_emb')[
                    0]
            end_emb = tf.concat(end_emb, 2)
            end_emb = tf.concat([end_emb, G], 2)
            # 8. double cross-entropy loss (actually applied after this function)
            # 8a. prepare logits
            # 8b. prepare argmax for output module

            # 8a. prepare logits
            # start_emb = [batch, length2, 10*embedding]
            # W_start = [10*embedding]
            # start_emb *w_start_index = start_scores
            # [batch, length2, 10*embedding] * [10*embedding] = [batch, length2]
            start_scores = tf.einsum('ijk,k->ij', start_emb, W_start)
            # end_emb = [batch, length2, 10*emb]
            # W_end = [10*emb]
            # end_emb *w_end_index = start_scores
            # [batch, length2, 10*emb] * [10*emb] = [batch, length2]
            end_scores = tf.einsum('ijk,k->ij', end_emb, W_end)

            # mask out-of-bounds slots by adding -1000
            support_mask = mask_for_lengths(support_length)
            start_scores = start_scores + support_mask
            end_scores = end_scores + support_mask

            start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                compute_spans(start_scores, end_scores, answer2support, is_eval, support2question)

            span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

            return start_scores, end_scores, span
