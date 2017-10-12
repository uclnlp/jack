from jack.readers.extractive_qa.shared import XQAPorts
from jack.tf_fun.xqa import xqa_min_crossentropy_loss
from jack.tf_fun.embedding import conv_char_embedding_alt
from jack.core.shared_resources import SharedResources
from jack.core import TFModelModule, Mapping, TensorPort, List, Sequence, Ports
from jack.util import tfutil
from jack.tf_fun.rnn import fused_birnn
from jack.tf_fun.highway import highway_network

import numpy as np
import tensorflow as tf
import logging
logger = logging.getLogger(__name__)


input_ports=[XQAPorts.emb_question, XQAPorts.question_length,
             XQAPorts.emb_support, XQAPorts.support_length,
             # char embedding inputs
             XQAPorts.unique_word_chars, XQAPorts.unique_word_char_length,
             XQAPorts.question_words2unique, XQAPorts.support_words2unique,
             # feature input
             XQAPorts.word_in_question,
             # optional input, provided only during training
             XQAPorts.correct_start_training, XQAPorts.answer2question_training,
             XQAPorts.keep_prob, XQAPorts.is_eval]
output_ports=[XQAPorts.start_scores, XQAPorts.end_scores,
              XQAPorts.span_prediction]
training_input_ports=[XQAPorts.start_scores, XQAPorts.end_scores,
                      XQAPorts.answer_span, XQAPorts.answer2question]
training_output_ports=[Ports.loss]

class AbstractExtractiveQA(TFModelModule):
    def __init__(self, shared_resources: SharedResources, sess=None):
        super(AbstractExtractiveQA, self).__init__(shared_resources)
        self.shared_resources = shared_resources

    #def __call__(self, batch: Mapping[TensorPort, np.ndarray],
    #             goal_ports: List[TensorPort] = None) -> Mapping[TensorPort, np.ndarray]:
    #    raise NotImplementedError("Not implemented yet...")

    @property
    def output_ports(self) -> Sequence[TensorPort]:
        return output_ports

    @property
    def input_ports(self) -> Sequence[TensorPort]:
        return input_ports

    @property
    def training_input_ports(self) -> Sequence[TensorPort]:
        return training_input_ports

    @property
    def training_output_ports(self) -> Sequence[TensorPort]:
        return training_output_ports

    def create_output(self, shared_resources: SharedResources, *tensors: tf.Tensor) -> Sequence[TensorPort]:
        return self.model_function(shared_resources, *tensors)

    def create_training_output(self, shared_resources: SharedResources, *tensors: tf.Tensor)\
            -> Sequence[TensorPort]:
        return xqa_min_crossentropy_loss(*tensors)


    def model_function(self, shared_vocab_config, emb_question, question_length,
                 emb_support, support_length,
                 unique_word_chars, unique_word_char_length,
                 question_words2unique, support_words2unique,
                 word_in_question,
                 correct_start, answer2question, keep_prob, is_eval):
        """
        fast_qa model
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
        raise NotImplementedError('Classes that inherit from AbstractExtractiveQA need to override model_ function!')

class BiDAF(AbstractExtractiveQA):
    def __init__(self, shared_resources):
        super(BiDAF, self).__init__(shared_resources)

    def model_function(self, shared_vocab_config, emb_question, question_length,
                 emb_support, support_length,
                 unique_word_chars, unique_word_char_length,
                 question_words2unique, support_words2unique,
                 word_in_question,
                 correct_start, answer2question, keep_prob, is_eval):

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
            batch_size = tf.shape(question_length)[0]
            max_question_length = tf.reduce_max(question_length)
            max_support_length = tf.reduce_max(support_length)
            support_mask = tfutil.mask_for_lengths(support_length, batch_size)
            question_binary_mask = tfutil.mask_for_lengths(question_length, batch_size, mask_right=False, value=1.0)
            beam_size = 1
            beam_size = tf.cond(is_eval, lambda: tf.constant(beam_size, tf.int32), lambda: tf.constant(1, tf.int32))

            input_size = shared_vocab_config.config["repr_dim_input"]
            size = shared_vocab_config.config["repr_dim"]
            with_char_embeddings = shared_vocab_config.config.get("with_char_embeddings", False)
            W = tf.get_variable("biattention_weight", [size*6])
            W_start_index = tf.get_variable("start_index_weight", [size*10])
            W_end_index = tf.get_variable("end_index_weight", [size*10])


            # 1. char embeddings + word embeddings
            # set shapes for inputs
            emb_question.set_shape([None, None, input_size])
            emb_support.set_shape([None, None, input_size])

            # 1. + 2a. + 2b. 2a. char embeddings + conv + max pooling
            # compute combined embeddings
            [char_emb_question, char_emb_support] = conv_char_embedding_alt(shared_vocab_config.config["char_vocab"],
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

            #emb_question = tf.slice(highway_question, [0, 0, 0], tf.stack([-1, max_question_length, -1]))
            #emb_support = tf.slice(all_embedded_hw, tf.stack([0, max_question_length, 0]), [-1, -1, -1])

            #emb_question.set_shape([None, None, size])
            #emb_support.set_shape([None, None, size])


            # 4. BiLSTM
            cell1 = tf.contrib.rnn.LSTMBlockFusedCell(size)
            encoded_question = fused_birnn(cell1, highway_question, question_length, dtype=tf.float32, time_major=False, scope='question_encoding')[0]
            encoded_question = tf.concat(encoded_question, 2)

            cell2 = tf.contrib.rnn.LSTMBlockFusedCell(size)
            encoded_support = fused_birnn(cell2, highway_support, support_length, dtype=tf.float32, time_major=False, scope='support_encoding')[0]
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
            features = tf.concat([support, question, question*support], 3)

            # 6. biattention
            # 6a. create matrix of question support attentions
            # features = [batch, length1, length2, 6*embeddings]
            # w = [6*embeddings]
            # S = attention matrix = [batch, length1, length2] 
            S = tf.einsum('ijkl,l->ijk', features, W)

            # S = [batch, length1, length2]
            # question to support attention
            # softmax -> [ batch, length1, length2] = att_question
            att_question = tf.nn.softmax(S, 2) # softmax over support
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
            G = tf.concat([encoded_support, question_weighted, encoded_support*question_weighted, encoded_support*support_weighted], 2)

            # 8. BiLSTM(G) = M
            # start_index = M
            cell3 = tf.contrib.rnn.LSTMBlockFusedCell(size)
            start_index = fused_birnn(cell3, G, support_length, dtype=tf.float32, time_major=False, scope='start_index')[0]
            start_index = tf.concat(start_index, 2)
            start_index = tf.concat([start_index, G], 2)
            # BiLSTM(M) = M^2 = end_index
            cell4 = tf.contrib.rnn.LSTMBlockFusedCell(size)
            end_index = fused_birnn(cell4, start_index, support_length, dtype=tf.float32, time_major=False, scope='end_index')[0]
            end_index = tf.concat(end_index, 2)
            end_index = tf.concat([end_index, G], 2)
            # 9. double cross-entropy loss
            # 9a. prepare logits
            # 9b. prepare top-k probabilities for beam search
            # 9c. prepare argmax for output module

            # start_index = [batch, length2, 10*embedding] 
            # W_start_index = [10*embedding]
            # start_index *w_start_index = start_scores
            # [batch, length2, 10*embedding] * [10*embedding] = [batch, length2] 
            # 9a. prepare logits
            start_scores = tf.einsum('ijk,k->ij', start_index, W_start_index)
            support_mask = tfutil.mask_for_lengths(support_length, batch_size)
            # add -1000 to out-of-bounds slots
            start_scores = start_scores + support_mask

            # probs are needed during beam search
            # 9b. prepare top-k probabilities for beam search
            start_probs = tf.nn.softmax(start_scores)
            predicted_start_probs, predicted_start_pointer = tf.nn.top_k(start_probs, beam_size)

            # use correct start during training, because p(end|start) should be optimized
            #print(predicted_start_pointer, answer2question)
            #predicted_start_pointer = tf.gather(predicted_start_pointer, answer2question)
            #predicted_start_probs = tf.gather(predicted_start_probs, answer2question)

            #start_pointer = tf.cond(is_eval, lambda: predicted_start_pointer, lambda: tf.expand_dims(correct_start, 1))

            ## flatten again
            #start_pointer = tf.reshape(start_pointer, [-1])
            #answer2questionwithbeam = tf.reshape(tf.tile(tf.expand_dims(answer2question, 1), tf.stack([1, beam_size])), [-1])

            #input_size = encoded_support.get_shape()[-1].value
            #support_states_flat = tf.reshape(encoded_support, [-1, input_size])

            #offsets = tf.cast(tf.range(0, batch_size) * max_support_length, dtype=tf.int32)
            #offsets = tf.gather(offsets, answer2questionwithbeam)
            #u_s = tf.gather(support_states_flat, start_pointer + offsets)

            #start_scores = tf.gather(start_scores, answer2questionwithbeam)

            #def mask_with_start(scores):
            #    return scores + tfutil.mask_for_lengths(tf.cast(start_pointer, tf.int32),
            #                                            batch_size * beam_size, tf.reduce_max(support_length),
            #                                            mask_right=False)

            # end_index = [batch, length2, 10*emb] 
            # W_end_index = [10*emb]
            # end_index *w_end_index = start_scores
            # [batch, length2, 10*emb] * [10*emb] = [batch, length2] 
            # 9a. prepare logits
            end_scores = tf.einsum('ijk,k->ij', end_index, W_end_index)
            # add -1000 to out-of-bounds slots
            end_scores = end_scores + support_mask
            #end_scores = tf.cond(is_eval, lambda: mask_with_start(end_scores), lambda: end_scores)

            # probs are needed during beam search
            end_probs = tf.nn.softmax(end_scores)


            predicted_end_probs, predicted_end_pointer = tf.nn.top_k(end_probs, 1)
            #predicted_end_probs = tf.reshape(predicted_end_probs, tf.stack([-1, beam_size]))
            #predicted_end_pointer = tf.reshape(predicted_end_pointer, tf.stack([-1, beam_size]))

            # 9c. prepare argmax for output module
            #predicted_idx = tf.cast(tf.argmax(predicted_start_probs * predicted_end_probs, 1), tf.int32)

            #predicted_start_pointer = tf.gather(predicted_start_pointer, predicted_idx)
            #predicted_end_pointer = tf.gather(predicted_end_pointer, predicted_idx)

            span = tf.concat([tf.expand_dims(predicted_start_pointer, 1), tf.expand_dims(predicted_end_pointer, 1)], 1)

            return start_scores, end_scores, span

