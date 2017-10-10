# -*- coding: utf-8 -*-

import tensorflow as tf


def get_by_index(tensor, index):
    """
    Args:
        tensor: [dim1 x dim2 x dim3] tensor
        index: [dim1] tensor of indices for dim2

    Returns:
        [dim1 x dim3] tensor
    """
    dim1, dim2, dim3 = tf.unstack(tf.shape(tensor))
    flat_index = tf.range(0, dim1) * dim2 + (index - 1)
    return tf.gather(tf.reshape(tensor, [-1, dim3]), flat_index)


def mask_for_lengths(lengths, batch_size=None, max_length=None, mask_right=True, value=-1000.0):
    """
    Creates a [batch_size x max_length] mask.

    Args:
        lengths: int32 1-dim tensor of batch_size lengths
        batch_size: int32 0-dim tensor or python int
        max_length: int32 0-dim tensor or python int
        mask_right: if True, everything before "lengths" becomes zero and the
            rest "value", else vice versa
        value: value for the mask

    Returns:
        [batch_size x max_length] mask of zeros and "value"s
    """
    mask = tf.sequence_mask(lengths, max_length, dtype=tf.float32)
    if mask_right:
        mask = 1.0 - mask
    mask *= value
    return mask
