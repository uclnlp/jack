# -*- coding: utf-8 -*-
import math

import tensorflow as tf

from jack.util.tf import segment, misc


def attention_softmax(attn_scores, length=None):
    if length is not None:
        attn_scores += misc.mask_for_lengths(length, tf.shape(attn_scores)[2])
    return tf.nn.softmax(attn_scores)


def apply_attention(attn_scores, states, length, is_self=False, with_sentinel=True, reuse=False, seq2_to_seq1=None):
    attn_scores += tf.expand_dims(misc.mask_for_lengths(length, tf.shape(attn_scores)[2]), 1)
    softmax = tf.nn.softmax if seq2_to_seq1 is None else lambda x: segment.segment_softmax(x, seq2_to_seq1)
    if is_self:
        # exclude attending to state itself
        attn_scores += tf.expand_dims(tf.diag(tf.fill([tf.shape(attn_scores)[1]], -1e6)), 0)
    if with_sentinel:
        with tf.variable_scope('sentinel', reuse=reuse):
            s = tf.get_variable('score', [1, 1, 1], tf.float32, tf.zeros_initializer())
        s = tf.tile(s, [tf.shape(attn_scores)[0], tf.shape(attn_scores)[1], 1])
        attn_probs = softmax(tf.concat([s, attn_scores], 2))
        attn_probs = attn_probs[:, :, 1:]
    else:
        attn_probs = softmax(attn_scores)
    attn_states = tf.einsum('abd,adc->abc', attn_probs, states)
    if seq2_to_seq1 is not None:
        attn_states = tf.unsorted_segment_sum(attn_states, seq2_to_seq1, tf.reduce_max(seq2_to_seq1) + 1)
    return attn_scores, attn_probs, attn_states


def dot_attention(seq1, seq2, len2, scaled=True, with_sentinel=True, seq2_to_seq1=None, **kwargs):
    query, key, value = _get_query_key_value(seq1, seq2, **kwargs)
    if seq2_to_seq1 is not None:
        query = tf.gather(query, seq2_to_seq1)
    attn_scores = tf.einsum('abc,adc->abd', query, key)
    if scaled:
        attn_scores /= tf.sqrt(float(query.get_shape()[-1].value))
    return apply_attention(attn_scores, value, len2, seq1 is seq2, with_sentinel, seq2_to_seq1=seq2_to_seq1)


def bilinear_attention(seq1, seq2, len2, scaled=True, with_sentinel=True, seq2_to_seq1=None, **kwargs):
    query, key, value = _get_query_key_value(seq1, seq2, **kwargs)
    if seq2_to_seq1 is not None:
        query = tf.gather(query, seq2_to_seq1)
    attn_scores = tf.einsum('abc,adc->abd', tf.layers.dense(query, key.get_shape()[-1].value, use_bias=False), key)
    attn_scores += tf.layers.dense(query, 1, use_bias=False)
    attn_scores += tf.transpose(tf.layers.dense(key, 1, use_bias=False), [0, 2, 1])
    if scaled:
        attn_scores /= math.sqrt(float(query.get_shape()[-1].value))
    return apply_attention(attn_scores, value, len2, seq1 is seq2, with_sentinel, seq2_to_seq1=seq2_to_seq1)


def diagonal_bilinear_attention(seq1, seq2, len2, scaled=True, with_sentinel=True, seq2_to_seq1=None, **kwargs):
    query, key, value = _get_query_key_value(seq1, seq2, **kwargs)
    if seq2_to_seq1 is not None:
        query = tf.gather(query, seq2_to_seq1)
    v = tf.get_variable('attn_weight', [1, 1, query.get_shape()[-1].value], tf.float32,
                        initializer=tf.ones_initializer())
    attn_scores = tf.einsum('abc,adc->abd', v * query, key)
    attn_scores += tf.layers.dense(query, 1, use_bias=False)
    attn_scores += tf.transpose(tf.layers.dense(key, 1, use_bias=False), [0, 2, 1])
    if scaled:
        attn_scores /= math.sqrt(float(query.get_shape()[-1].value))
    return apply_attention(attn_scores, value, len2, seq1 is seq2, with_sentinel, seq2_to_seq1=seq2_to_seq1)


def mlp_attention(hidden_dim, seq1, seq2, len2, activation=tf.nn.relu, with_sentinel=True, seq2_to_seq1=None, **kwargs):
    query, key, value = _get_query_key_value(seq1, seq2, **kwargs)
    if seq2_to_seq1 is not None:
        query = tf.gather(query, seq2_to_seq1)
    hidden1 = tf.layers.dense(query, hidden_dim)
    hidden2 = tf.layers.dense(key, hidden_dim)
    hidden = tf.tile(tf.expand_dims(hidden1, 2), [1, 1, tf.shape(key)[1], -1]) + tf.expand_dims(hidden2, 1)
    attn_scores = tf.layers.dense(activation(hidden), 1, use_bias=with_sentinel)
    return apply_attention(tf.squeeze(attn_scores, 3), value, len2, seq1 is seq2, with_sentinel,
                           seq2_to_seq1=seq2_to_seq1)


def _get_query_key_value(seq1, seq2, key_value_attn=False, repr_dim=None, **kwargs):
    repr_dim = repr_dim or seq2.get_shape()[-1].value
    if key_value_attn:
        with tf.variable_scope('key_value_projection') as vs:
            key = tf.layers.dense(seq2, repr_dim, name='key')
            value = tf.layers.dense(seq2, repr_dim, name='value')
            query = tf.layers.dense(seq1, repr_dim, name='query')
        return query, key, value
    else:
        return seq1, seq2, seq2


def coattention(seq1, len1, seq2, len2, scaled=True, with_sentinel=True, attn_fn=dot_attention, seq2_to_seq1=None):
    attn_scores, attn_probs1, attn_states1 = attn_fn(seq1, seq2, len2, scaled, with_sentinel, seq2_to_seq1)
    _, _, attn_states2 = apply_attention(
        tf.transpose(attn_scores, [0, 2, 1]), seq1, len1, seq1 is seq2, with_sentinel, reuse=True)
    co_attn_state = tf.einsum('abd,adc->abc', attn_probs1, attn_states2)

    return attn_scores, attn_probs1, attn_states1, attn_states2, co_attn_state


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
