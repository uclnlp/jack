# -*- coding: utf-8 -*-

import tensorflow as tf

from jack.tfutil import misc


def attention_softmax(attn_scores, length=None):
    attn_scores += misc.mask_for_lengths(length, tf.shape(attn_scores)[2])
    return tf.nn.softmax(attn_scores)


def apply_attention(attn_scores, states, length):
    attn_scores += tf.expand_dims(misc.mask_for_lengths(length, tf.shape(attn_scores)[2]), 1)
    attn_probs = tf.nn.softmax(attn_scores)
    attn_states = tf.einsum('abd,adc->abc', attn_probs, states)
    return attn_scores, attn_probs, attn_states


def dot_attention(seq1, seq2, len2, scaled=True):
    attn_scores = tf.einsum('abc,adc->abd', seq1, seq2)
    if scaled:
        attn_scores /= tf.sqrt(float(seq1.get_shape()[-1].value))
    return apply_attention(attn_scores, seq2, len2)


def bilinear_attention(seq1, seq2, len2, scaled=True):
    seq1 = tf.layers.dense(seq1, seq2.get_shape()[-1].value, scaled)
    return dot_attention(seq1, seq2, len2)


def diagonal_bilinear_attention(seq1, seq2, len2, scaled=True):
    v = tf.get_variable('attn_weight', [1, 1, seq2.get_shape()[-1].value], tf.float32,
                        initializer=tf.ones_initializer())
    return dot_attention(seq1 * v, seq2, len2)


def mlp_attention(hidden_dim, seq1, seq2, len2, activation=tf.nn.relu):
    hidden1 = tf.layers.dense(seq1, hidden_dim)
    hidden2 = tf.layers.dense(seq2, hidden_dim)
    hidden = tf.tile(tf.expand_dims(hidden1, 2), [1, 1, tf.shape(seq2)[1], -1]) + tf.expand_dims(hidden2, 1)
    attn_scores = tf.layers.dense(activation(hidden), 1, use_bias=False)
    return apply_attention(tf.squeeze(attn_scores, 3), seq2, len2)


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
