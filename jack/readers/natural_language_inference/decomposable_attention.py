# -*- coding: utf-8 -*-

import logging
from abc import abstractmethod

import numpy as np
import tensorflow as tf

from jack.readers.multiple_choice.shared import AbstractSingleSupportFixedClassModel
from jack.tfutil.activations import prelu
from jack.tfutil.attention import attention_softmax3d
from jack.tfutil.masking import mask_3d

logger = logging.getLogger(__name__)


class DecomposableAttentionModel(AbstractSingleSupportFixedClassModel):
    def forward_pass(self, shared_resources,
                     question, question_length,
                     support, support_length,
                     num_classes):
        # final states_fw_bw dimensions:
        # [[[batch, output dim], [batch, output_dim]]
        question_embedding = tf.nn.embedding_lookup(self.question_embedding_matrix, question)
        support_embedding = tf.nn.embedding_lookup(self.support_embedding_matrix, support)

        model_kwargs = {
            'sequence1': question_embedding,
            'sequence1_length': question_length,
            'sequence2': support_embedding,
            'sequence2_length': support_length,
            'representation_size': 200,
            'dropout_keep_prob': 1.0 - shared_resources.config.get('dropout', 0),
            'use_masking': True,
            'prepend_null_token': True
        }

        model = FeedForwardDAMP(**model_kwargs)
        logits = model()
        return logits


class BaseDecomposableAttentionModel:
    @abstractmethod
    def _transform_input(self, sequence, reuse=False):
        raise NotImplementedError

    @abstractmethod
    def _transform_attend(self, sequence, reuse=False):
        raise NotImplementedError

    @abstractmethod
    def _transform_compare(self, sequence, reuse=False):
        raise NotImplementedError

    @abstractmethod
    def _transform_aggregate(self, v1_v2, reuse=False):
        raise NotImplementedError

    def __init__(self,
                 sequence1, sequence1_length, sequence2, sequence2_length,
                 nb_classes=3, reuse=False,
                 use_masking=False, prepend_null_token=False, *args, **kwargs):
        self.nb_classes = nb_classes

        self.sequence1 = sequence1
        self.sequence1_length = sequence1_length

        self.sequence2 = sequence2
        self.sequence2_length = sequence2_length

        self.reuse = reuse

        batch_size = tf.shape(self.sequence1)[0]

        embedding1_size = self.sequence1.get_shape()[-1].value
        embedding2_size = self.sequence2.get_shape()[-1].value

        assert embedding1_size == embedding2_size

        # [batch_size, time_steps, embedding_size] -> [batch_size, time_steps, representation_size]
        self.transformed_sequence1 = self._transform_input(self.sequence1, reuse=self.reuse)

        # [batch_size, time_steps, embedding_size] -> [batch_size, time_steps, representation_size]
        self.transformed_sequence2 = self._transform_input(self.sequence2, reuse=True)

        self.transformed_sequence1_length = self.sequence1_length
        self.transformed_sequence2_length = self.sequence2_length

        self.null_token_embedding = None

        if prepend_null_token:
            with tf.variable_scope('null', reuse=self.reuse) as _:
                # [1, 1, embedding_size]
                self.null_token_embedding = tf.get_variable('null_embedding',
                                                            shape=[1, embedding1_size],
                                                            initializer=tf.random_normal_initializer(0.0, 1.0))

            # [1, 1, representation_size]
            transformed_null_token_embedding = self._transform_input(self.null_token_embedding, reuse=True)

            # [batch_size, 1, representation_size]
            tiled_null_token_embedding = tf.tile(input=tf.expand_dims(transformed_null_token_embedding, axis=0),
                                                 multiples=[batch_size, 1, 1])

            # [batch_size, time_steps + 1, representation_size]
            self.transformed_sequence1 = tf.concat(values=[tiled_null_token_embedding, self.transformed_sequence1],
                                                   axis=1)

            # [batch_size, time_steps + 1, representation_size]
            self.transformed_sequence2 = tf.concat(values=[tiled_null_token_embedding, self.transformed_sequence2],
                                                   axis=1)

            self.transformed_sequence1_length += 1
            self.transformed_sequence2_length += 1

        logger.info('Building the Attend graph ..')

        self.raw_attentions = None
        self.attention_sentence1 = self.attention_sentence2 = None

        # tensors with shape (batch_size, time_steps, num_units)
        self.alpha, self.beta = self.attend(self.transformed_sequence1, self.transformed_sequence2,
                                            sequence1_lengths=self.transformed_sequence1_length,
                                            sequence2_lengths=self.transformed_sequence2_length,
                                            use_masking=use_masking, reuse=self.reuse)

        logger.info('Building the Compare graph ..')

        # tensor with shape (batch_size, time_steps, num_units)
        self.v1 = self.compare(self.transformed_sequence1, self.beta, reuse=self.reuse)

        # tensor with shape (batch_size, time_steps, num_units)
        self.v2 = self.compare(self.transformed_sequence2, self.alpha, reuse=True)

        logger.info('Building the Aggregate graph ..')
        self.logits = self.aggregate(self.v1, self.v2, self.nb_classes,
                                     v1_lengths=self.transformed_sequence1_length,
                                     v2_lengths=self.transformed_sequence2_length,
                                     use_masking=use_masking, reuse=self.reuse)

    def __call__(self):
            return self.logits

    def attend(self, sequence1, sequence2,
               sequence1_lengths=None, sequence2_lengths=None, use_masking=False, reuse=False):
        """
        Attend phase.

        Args:
            sequence1: tensor with shape (batch_size, time_steps, num_units)
            sequence2: tensor with shape (batch_size, time_steps, num_units)
            sequence1_lengths: time_steps in sequence1
            sequence2_lengths: time_steps in sequence2
            use_masking: use masking
            reuse: reuse variables

        Returns:
            two tensors with shape (batch_size, time_steps, num_units)
        """
        with tf.variable_scope('attend') as _:
            # tensor with shape (batch_size, time_steps, num_units)
            transformed_sequence1 = self._transform_attend(sequence1, reuse)

            # tensor with shape (batch_size, time_steps, num_units)
            transformed_sequence2 = self._transform_attend(sequence2, True)

            # tensor with shape (batch_size, time_steps, time_steps)
            self.raw_attentions = tf.matmul(transformed_sequence1, tf.transpose(transformed_sequence2, [0, 2, 1]))

            masked_raw_attentions = self.raw_attentions
            if use_masking:
                masked_raw_attentions = mask_3d(sequences=masked_raw_attentions,
                                                sequence_lengths=sequence2_lengths,
                                                mask_value=- np.inf, dimension=2)
            self.attention_sentence1 = attention_softmax3d(masked_raw_attentions)

            # tensor with shape (batch_size, time_steps, time_steps)
            attention_transposed = tf.transpose(self.raw_attentions, [0, 2, 1])
            masked_attention_transposed = attention_transposed
            if use_masking:
                masked_attention_transposed = mask_3d(sequences=masked_attention_transposed,
                                                      sequence_lengths=sequence1_lengths,
                                                      mask_value=- np.inf, dimension=2)
            self.attention_sentence2 = attention_softmax3d(masked_attention_transposed)

            # tensors with shape (batch_size, time_steps, num_units)
            alpha = tf.matmul(self.attention_sentence2, sequence1, name='alpha')
            beta = tf.matmul(self.attention_sentence1, sequence2, name='beta')
            return alpha, beta

    def compare(self, sentence, soft_alignment, reuse=False):
        """
        Compare phase.

        Args:
            sentence: tensor with shape (batch_size, time_steps, num_units)
            soft_alignment: tensor with shape (batch_size, time_steps, num_units)
            reuse: reuse variables

        Returns:
            tensor with shape (batch_size, time_steps, num_units)
        """
        # tensor with shape (batch, time_steps, num_units)
        sentence_and_alignment = tf.concat(axis=2, values=[sentence, soft_alignment])
        transformed_sentence_and_alignment = self._transform_compare(sentence_and_alignment, reuse=reuse)
        return transformed_sentence_and_alignment

    def aggregate(self, v1, v2, num_classes,
                  v1_lengths=None, v2_lengths=None, use_masking=False, reuse=False):
        """
        Aggregate phase.

        Args:
            v1: tensor with shape (batch_size, time_steps, num_units)
            v2: tensor with shape (batch_size, time_steps, num_units)
            num_classes: number of output units
            v1_lengths: time_steps in v1
            v2_lengths: time_steps in v2
            use_masking: use masking
            reuse: reuse variables
        """
        with tf.variable_scope('aggregate', reuse=reuse) as _:
            if use_masking:
                v1 = mask_3d(sequences=v1, sequence_lengths=v1_lengths, mask_value=0, dimension=1)
                v2 = mask_3d(sequences=v2, sequence_lengths=v2_lengths, mask_value=0, dimension=1)

            v1_sum, v2_sum = tf.reduce_sum(v1, [1]), tf.reduce_sum(v2, [1])

            v1_v2 = tf.concat(axis=1, values=[v1_sum, v2_sum])
            transformed_v1_v2 = self._transform_aggregate(v1_v2, reuse=reuse)

            logits = tf.contrib.layers.fully_connected(inputs=transformed_v1_v2,
                                                       num_outputs=num_classes,
                                                       weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                       biases_initializer=tf.zeros_initializer(),
                                                       activation_fn=None)
        return logits


class FeedForwardDAM(BaseDecomposableAttentionModel):
    def __init__(self, representation_size=200, dropout_keep_prob=1.0, *args, **kwargs):
        self.representation_size = representation_size
        self.dropout_keep_prob = dropout_keep_prob
        super().__init__(*args, **kwargs)

    def _transform_input(self, sequence, reuse=False):
        with tf.variable_scope('transform_embeddings', reuse=reuse):
            projection = tf.contrib.layers.fully_connected(inputs=sequence, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           activation_fn=None)
        return projection

    def _transform_attend(self, sequence, reuse=False):
        with tf.variable_scope('transform_attend', reuse=reuse):
            projection = tf.nn.dropout(sequence, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu)
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu)
        return projection

    def _transform_compare(self, sequence, reuse=False):
        with tf.variable_scope('transform_compare', reuse=reuse):
            projection = tf.nn.dropout(sequence, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu)
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu)
        return projection

    def _transform_aggregate(self, v1_v2, reuse=False):
        with tf.variable_scope('transform_aggregate', reuse=reuse):
            projection = tf.nn.dropout(v1_v2, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu)
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer(),
                                                           activation_fn=tf.nn.relu)
        return projection


class FeedForwardDAMP(BaseDecomposableAttentionModel):
    def __init__(self, representation_size=200, dropout_keep_prob=1.0, *args, **kwargs):
        self.representation_size = representation_size
        self.dropout_keep_prob = dropout_keep_prob
        super().__init__(*args, **kwargs)

    def _transform_input(self, sequence, reuse=False):
        with tf.variable_scope('transform_embeddings', reuse=reuse) as _:
            projection = tf.contrib.layers.fully_connected(inputs=sequence, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           activation_fn=None)
        return projection

    def _transform_attend(self, sequence, reuse=False):
        with tf.variable_scope('transform_attend', reuse=reuse) as _:
            projection = tf.nn.dropout(sequence, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = prelu(projection, name='1')
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = prelu(projection, name='2')
        return projection

    def _transform_compare(self, sequence, reuse=False):
        with tf.variable_scope('transform_compare', reuse=reuse) as _:
            projection = tf.nn.dropout(sequence, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = prelu(projection, name='1')
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = prelu(projection, name='2')
        return projection

    def _transform_aggregate(self, v1_v2, reuse=False):
        with tf.variable_scope('transform_aggregate', reuse=reuse) as _:
            projection = tf.nn.dropout(v1_v2, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = prelu(projection, name='1')
            projection = tf.nn.dropout(projection, keep_prob=self.dropout_keep_prob)
            projection = tf.contrib.layers.fully_connected(inputs=projection, num_outputs=self.representation_size,
                                                           weights_initializer=tf.random_normal_initializer(0.0, 0.01),
                                                           biases_initializer=tf.zeros_initializer())
            projection = prelu(projection, name='2')
        return projection
