"""
This file contains FastQA specific modules and ports
"""

from jack.core import *
from jack.readers.extractive_qa.tensorflow.abstract_model import AbstractXQAModelModule
from jack.readers.extractive_qa.tensorflow.answer_layer import conditional_answer_layer, bilinear_answer_layer
from jack.util.tf import misc
from jack.util.tf.embedding import conv_char_embedding
from jack.util.tf.highway import highway_network
from jack.util.tf.sequence_encoder import encoder


class FastQAModule(AbstractXQAModelModule):
    def set_topk(self, k):
        self._topk_assign(k)

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        with tf.variable_scope("fast_qa", initializer=tf.contrib.layers.xavier_initializer()):
            # Some helpers
            batch_size = tf.shape(tensors.question_length)[0]
            max_question_length = tf.reduce_max(tensors.question_length)
            support_mask = misc.mask_for_lengths(tensors.support_length)

            input_size = shared_resources.embeddings.shape[-1]
            size = shared_resources.config["repr_dim"]
            with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)

            # set shapes for inputs
            tensors.emb_question.set_shape([None, None, input_size])
            tensors.emb_support.set_shape([None, None, input_size])

            emb_question = tensors.emb_question
            emb_support = tensors.emb_support
            if with_char_embeddings:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    len(shared_resources.char_vocab), size, tensors.word_chars, tensors.word_char_length,
                    [tensors.question_batch_words, tensors.support_batch_words])

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

            wiq_w = tf.matmul(tf.gather(emb_question * v_wiqw, tensors.support2question), emb_support, adjoint_b=True)
            wiq_w = wiq_w + tf.expand_dims(support_mask, 1)

            question_binary_mask = tf.gather(tf.sequence_mask(tensors.question_length, dtype=tf.float32),
                                             tensors.support2question)
            wiq_w = tf.reduce_sum(tf.nn.softmax(wiq_w) * tf.expand_dims(question_binary_mask, 2), [1])

            # [B, L , 2]
            support_features = tf.stack([tensors.word_in_question, wiq_w], 2)

            # highway layer to allow for interaction between concatenated embeddings
            if with_char_embeddings:
                with tf.variable_scope("char_embeddings") as vs:
                    emb_question = tf.layers.dense(emb_question, size, name="embeddings_projection")
                    emb_question = highway_network(emb_question, 1)
                    vs.reuse_variables()
                    emb_support = tf.layers.dense(emb_support, size, name="embeddings_projection")
                    emb_support = highway_network(emb_support, 1)

            keep_prob = 1.0 - shared_resources.config.get("dropout", 0.0)
            emb_question, emb_support = tf.cond(
                tensors.is_eval,
                lambda: (emb_question, emb_support),
                lambda: (tf.nn.dropout(emb_question, keep_prob, noise_shape=[1, 1, emb_question.get_shape()[-1].value]),
                         tf.nn.dropout(emb_support, keep_prob, noise_shape=[1, 1, emb_question.get_shape()[-1].value]))
            )

            # extend embeddings with features
            emb_question_ext = tf.concat([emb_question, question_features], 2)
            emb_support_ext = tf.concat([emb_support, support_features], 2)

            # encode question and support
            encoder_type = shared_resources.config.get('encoder', 'lstm').lower()
            if encoder_type in ['lstm', 'sru', 'gru']:
                size = size + 2 if encoder_type == 'sru' else size  # to allow for use of residual in SRU
                encoded_question = encoder(emb_question_ext, tensors.question_length, size, module=encoder_type)
                encoded_support = encoder(emb_support_ext, tensors.support_length, size, module=encoder_type,
                                          reuse=True)
                projection_initializer = tf.constant_initializer(np.concatenate([np.eye(size), np.eye(size)]))
                encoded_question = tf.layers.dense(encoded_question, size, tf.tanh, use_bias=False,
                                                   kernel_initializer=projection_initializer,
                                                   name='projection_q')
                encoded_support = tf.layers.dense(encoded_support, size, tf.tanh, use_bias=False,
                                                  kernel_initializer=projection_initializer, name='projection_s')
            else:
                raise ValueError("Only rnn ('lstm', 'sru', 'gru') encoder allowed for FastQA!")

            answer_layer = shared_resources.config.get('answer_layer', 'conditional').lower()

            topk = tf.get_variable(
                'topk', initializer=shared_resources.config.get('topk', 1), dtype=tf.int32, trainable=False)
            topk_p = tf.placeholder(tf.int32, [], 'beam_size_setter')
            topk_assign = topk.assign(topk_p)
            self._topk_assign = lambda k: self.tf_session.run(topk_assign, {topk_p: k})

            if answer_layer == 'conditional':
                start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                    conditional_answer_layer(size, encoded_question, tensors.question_length, encoded_support,
                                             tensors.support_length,
                                             tensors.correct_start, tensors.support2question, tensors.answer2support,
                                             tensors.is_eval,
                                             topk=topk,
                                             max_span_size=shared_resources.config.get("max_span_size", 10000))
            elif answer_layer == 'conditional_bilinear':
                start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                    conditional_answer_layer(size, encoded_question, tensors.question_length, encoded_support,
                                             tensors.support_length,
                                             tensors.correct_start, tensors.support2question, tensors.answer2support,
                                             tensors.is_eval,
                                             topk=topk,
                                             max_span_size=shared_resources.config.get("max_span_size", 10000),
                                             bilinear=True)
            elif answer_layer == 'bilinear':
                start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                    bilinear_answer_layer(size, encoded_question, tensors.question_length, encoded_support,
                                          tensors.support_length,
                                          tensors.support2question, tensors.answer2support, tensors.is_eval,
                                          topk=topk,
                                          max_span_size=shared_resources.config.get("max_span_size", 10000))
            else:
                raise ValueError

            span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

            return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))
