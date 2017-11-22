import logging

import tensorflow as tf

from jack.readers.extractive_qa.shared import AbstractXQAModelModule
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.highway import highway_network
from jack.tfutil.misc import mask_for_lengths
from jack.tfutil.rnn import fused_birnn

logger = logging.getLogger(__name__)


class BiDAF(AbstractXQAModelModule):
    def create_output(self, shared_resources, emb_question, question_length,
                      emb_support, support_length, support2question,
                      unique_word_chars, unique_word_char_length,
                      question_words2unique, support_words2unique,
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
            W_start_index = tf.get_variable("start_index_weight", [size * 10])
            W_end_index = tf.get_variable("end_index_weight", [size * 10])

            # 1. char embeddings + word embeddings
            # set shapes for inputs
            emb_question.set_shape([None, None, input_size])
            emb_support.set_shape([None, None, input_size])

            # 1. + 2a. + 2b. 2a. char embeddings + conv + max pooling
            if with_char_embeddings:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(len(shared_resources.char_vocab),
                                                                            size,
                                                                            unique_word_chars, unique_word_char_length,
                                                                            [question_words2unique,
                                                                             support_words2unique])
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
            encoded_question = tf.concat(encoded_question, 2)

            cell2 = tf.contrib.rnn.LSTMBlockFusedCell(size)
            encoded_support = fused_birnn(cell2, highway_support, support_length, dtype=tf.float32, time_major=False,
                                          scope='support_encoding')[0]
            encoded_support = tf.concat(encoded_support, 2)

            # 6. biattention alpha(U, H) = S
            # S = W^T*[H; U; H*U]
            # question = U = [batch, 2*embedding, length1]
            # support = H = [batch, 2*embedding, length2]
            # -> expand with features

            # we want to get from [length 1] and [length 2] to [length1, length2] and [length1, length2]
            # we do that with
            # (a) expand dim
            # [batch, L2, 2*embedding ] -> [batch, 1, L2 2*embedding]
            support = tf.expand_dims(encoded_support, 1)
            # [batch, L1, 2*embedding] -> [batch, L1, 1, 2*embedding]
            question = tf.expand_dims(encoded_question, 2)
            # (b) tile with the other dimension
            support = tf.tile(support, [1, max_question_length, 1, 1])
            question = tf.tile(question, [1, 1, max_support_length, 1])

            # 5. cat
            # question = U = [batch, length1, length2, 2*embeddings]
            # support = H = [batch, length1, length2, 2*embeddings]
            # S = W^T*[H; U; H*U]
            features = tf.concat([support, question, question * support], 3)

            # 6. biattention
            # 6a. create matrix of question support attentions
            # features = [batch, length1, length2, 6*embeddings]
            # w = [6*embeddings]
            # S = attention matrix = [batch, length1, length2]
            S = tf.einsum('ijkl,l->ijk', features, W)

            # S = [batch, length1, length2]
            # question to support attention
            # softmax -> [ batch, length1, length2] = att_question
            att_question = tf.nn.softmax(S, 2)  # softmax over support
            # weighted =  [batch, length1, length2] * [batch, length1, length2, 2*embedding] -> [batch, length2, 2*embedding]
            question_weighted = tf.einsum('ijk,ijkl->ikl', att_question, question)

            # support to question attention
            # 1. filter important context words with max
            # 2. softmax over question to get the question words which are most relevant for the most relevant context words
            # max(S) = [batch, length1, length2] -> [ batch, length1] = most important context
            max_support = tf.reduce_max(S, 2)
            # softmax over question -> [batch, length1]
            support_attention = tf.nn.softmax(max_support, 1)
            # support attention * support = weighted support
            # [batch, length1] * [batch, length1, length2, 2*embedding] = [batch, 2*embedding]
            support_weighted = tf.einsum('ij,ijkl->il', support_attention, support)
            # tile to have the same dimension
            # [batch, 2*embedding] -> [batch, length2, 2*embedding]
            support_weighted = tf.expand_dims(support_weighted, 1)
            support_weighted = tf.tile(support_weighted, [1, max_support_length, 1])

            # 6b. generate feature matrix
            # G(support, weighted question, weighted support)  = G(h, *u, *h) = [h, *u, mul(h, *u), mul(h, h*)] = [batch, length2, embedding*8]
            G = tf.concat([encoded_support, question_weighted, encoded_support * question_weighted,
                           encoded_support * support_weighted], 2)

            # 8. BiLSTM(G) = M
            # start_index = M
            cell3 = tf.contrib.rnn.LSTMBlockFusedCell(size)
            start_index = \
                fused_birnn(cell3, G, support_length, dtype=tf.float32, time_major=False, scope='start_index')[0]
            start_index = tf.concat(start_index, 2)
            start_index = tf.concat([start_index, G], 2)
            # BiLSTM(M) = M^2 = end_index
            cell4 = tf.contrib.rnn.LSTMBlockFusedCell(size)
            end_index = \
                fused_birnn(cell4, start_index, support_length, dtype=tf.float32, time_major=False, scope='end_index')[
                    0]
            end_index = tf.concat(end_index, 2)
            end_index = tf.concat([end_index, G], 2)
            # 9. double cross-entropy loss (actually applied after this function)
            # 9a. prepare logits
            # 9b. prepare argmax for output module

            # 9a. prepare logits
            # start_index = [batch, length2, 10*embedding]
            # W_start_index = [10*embedding]
            # start_index *w_start_index = start_scores
            # [batch, length2, 10*embedding] * [10*embedding] = [batch, length2]
            start_scores = tf.einsum('ijk,k->ij', start_index, W_start_index)
            # end_index = [batch, length2, 10*emb]
            # W_end_index = [10*emb]
            # end_index *w_end_index = start_scores
            # [batch, length2, 10*emb] * [10*emb] = [batch, length2]
            end_scores = tf.einsum('ijk,k->ij', end_index, W_end_index)

            # mask out-of-bounds slots by adding -1000
            support_mask = mask_for_lengths(support_length)
            start_scores = start_scores + support_mask
            end_scores = end_scores + support_mask

            # 9b. prepare argmax for output module
            predicted_start_pointer = tf.argmax(start_scores, 1, output_type=tf.int32)
            predicted_end_pointer = tf.argmax(end_scores, 1, output_type=tf.int32)

            # can only deal with single doc setups
            doc_idx = tf.zeros_like(predicted_start_pointer, tf.int32)
            span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

            return start_scores, end_scores, span
