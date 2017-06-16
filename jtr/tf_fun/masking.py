# -*- coding: utf-8 -*-

import tensorflow as tf


def mask_3d(sequences, sequence_lengths, mask_value, dimension=2):
    """
    Given a batch of matrices, each with shape m x n, mask the values in each
    row after the positions indicated in sentence_sizes.
    This function is supposed to mask the last columns in the raw attention
    matrix (e_{i, j}) in cases where the sentence2 is smaller than the
    maximum.
    
    Args:
        sequences: tensor with shape (batch_size, m, n)
        sequence_lengths: tensor with shape (batch_size) containing the sentence sizes that
           should be limited
        mask_value: scalar value to assign to items after sentence size
        dimension: over which dimension to mask values
    Returns:
        A tensor with the same shape as `values`
    """
    if dimension == 1:
        sequences = tf.transpose(sequences, [0, 2, 1])
    time_steps1, time_steps2 = tf.shape(sequences)[1], tf.shape(sequences)[2]
    ones = tf.ones_like(sequences, dtype=tf.int32)
    pad_values = mask_value * tf.cast(ones, tf.float32)
    mask = tf.sequence_mask(sequence_lengths, time_steps2)
    # mask is (batch_size, sentence2_size). we have to tile it for 3d
    mask3d = tf.tile(tf.expand_dims(mask, 1), (1, time_steps1, 1))
    masked = tf.where(mask3d, sequences, pad_values)
    return tf.transpose(masked, [0, 2, 1]) if dimension == 1 else masked
