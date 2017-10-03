# -*- coding: utf-8 -*-

import tensorflow as tf


def attention_softmax3d(values):
    """
    Performs a softmax over the attention values.
    Args:
        values: tensor with shape (batch_size, time_steps, time_steps)
    Returns:
        tensor with shape (batch_size, time_steps, time_steps)
    """
    original_shape = tf.shape(values)
    # tensor with shape (batch_size * time_steps, time_steps)
    reshaped_values = tf.reshape(tensor=values, shape=[-1, original_shape[2]])
    # tensor with shape (batch_size * time_steps, time_steps)
    softmax_reshaped_values = tf.nn.softmax(reshaped_values)
    # tensor with shape (batch_size, time_steps, time_steps)
    return tf.reshape(softmax_reshaped_values, original_shape)


def distance_biases(time_steps, window_size=10, reuse=False):
    """
    Return a 2-d tensor with the values of the distance biases to be applied
    on the intra-attention matrix of size sentence_size

    Args:
        time_steps: tensor scalar
        window_size: window size
        reuse: reuse variables
    Returns:
        2-d tensor (time_steps, time_steps)
    """
    with tf.variable_scope('distance-bias', reuse=reuse):
        # this is d_{i-j}
        distance_bias = tf.get_variable('dist_bias', [window_size], initializer=tf.zeros_initializer())
        r = tf.range(0, time_steps)
        r_matrix = tf.tile(tf.reshape(r, [1, -1]), tf.stack([time_steps, 1]))
        raw_idxs = r_matrix - tf.reshape(r, [-1, 1])
        clipped_idxs = tf.clip_by_value(raw_idxs, 0, window_size - 1)
        values = tf.nn.embedding_lookup(distance_bias, clipped_idxs)
    return values


def intra_attention(sequence, reuse=False):
    """
    Compute the intra attention of a sentence. It returns a concatenation
    of the original sentence with its attended output.

    Args:
        sequence: tensor in shape (batch, time_steps, num_units)
        reuse: reuse variables
    Returns:
        a tensor in shape (batch, time_steps, 2*num_units)
    """
    time_steps = tf.shape(sequence)[1]
    with tf.variable_scope('intra-attention') as _:
        # this is F_intra in the paper
        # f_intra1 is (batch, time_steps, num_units) and
        # f_intra1_t is (batch, num_units, time_steps)
        f_intra_t = tf.transpose(sequence, [0, 2, 1])

        # these are f_ij
        # raw_attentions is (batch, time_steps, time_steps)
        raw_attentions = tf.matmul(sequence, f_intra_t)

        # bias has shape (time_steps, time_steps)
        bias = distance_biases(time_steps, reuse=reuse)

        # bias is broadcast along batches
        raw_attentions += bias
        attentions = attention_softmax3d(raw_attentions)
        attended = tf.matmul(attentions, sequence)
    return tf.concat(axis=2, values=[sequence, attended])
