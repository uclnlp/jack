# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


class CompactifyCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, max_compact_length, input_dim, zero_result=None):
        # zero_result: [batch_size, max_compact_length, input_dim]
        self._max_compact_length = max_compact_length
        self._input_dim = input_dim
        self._shift_matrix = np.zeros((max_compact_length, max_compact_length))
        for i in range(0, max_compact_length):
            self._shift_matrix[i, (i + 1) % max_compact_length] = 1.0

        self._zero_result = zero_result

    def __call__(self, inputs, state, scope=None):
        result, counter = state
        result_matrix = tf.reshape(result, (-1, self._max_compact_length, self._input_dim))
        # result matrix # [batch_size, input_dim, max_compact_length]
        # counter [batch_size, max_compact_length]
        # inputs [batch_size, input_dim + 1]
        # mask = [batch_size, 1]
        mask = inputs[:, 0:1]  # [batch_size, 1]
        input_tokens = inputs[:, 1:]  # [batch_size, input_dim]

        input_at_counter = tf.expand_dims(input_tokens, 1) * tf.expand_dims(counter, 2)
        # [batch_size, max_compact_length, input_dim]
        expanded_mask = tf.expand_dims(mask, 1)

        #         new_result_matrix = mask * (result_matrix + input_at_counter) + (1.0 - mask) * result_matrix
        new_result_matrix = expanded_mask * (result_matrix + input_at_counter) + (1.0 - expanded_mask) * result_matrix
        new_counter = mask * tf.matmul(counter, tf.constant(self._shift_matrix, dtype=tf.float32)) + (
                                                                                                         1.0 - mask) * counter
        #         new_counter = tf.matmul(counter,tf.constant(self._shift_matrix,dtype=tf.float32))
        #         new_counter = counter

        new_result = tf.reshape(new_result_matrix, tf.shape(result))
        #         new_result = tf.reshape(input_at_counter, tf.shape(result))

        return new_result, (new_result, new_counter)

    def zero_state(self, batch_size, dtype):
        zero_result = tf.zeros((batch_size, self._input_dim * self._max_compact_length)) \
            if self._zero_result is None else tf.reshape(self._zero_result,
                                                         (batch_size, self._input_dim * self._max_compact_length))
        zero_counter = tf.concat([tf.ones((batch_size, 1)),
            tf.zeros((batch_size, self._max_compact_length - 1))], 1)
        return (zero_result, zero_counter)

    @property
    def state_size(self):
        return self._input_dim * self._max_compact_length, self._max_compact_length

    @property
    def output_size(self):
        return self._input_dim * self._max_compact_length


def to_inputs(inputs, masks):
    # inputs [batch_size, length, input_dim]
    # mask [batch_size, length]
    expanded_mask = tf.expand_dims(masks, 2)  # [batch,size, max_length]
    return tf.concat([expanded_mask, inputs], 2)
