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
    if max_length is None:
        max_length = tf.reduce_max(lengths)
    if batch_size is None:
        batch_size = tf.shape(lengths)[0]
    # [batch_size x max_length]
    mask = tf.reshape(tf.tile(tf.range(0, max_length), [batch_size]), tf.stack([batch_size, -1]))
    if mask_right:
        mask = tf.greater_equal(mask, tf.expand_dims(lengths, 1))
    else:
        mask = tf.less(mask, tf.expand_dims(lengths, 1))
    mask = tf.cast(mask, tf.float32) * value
    return mask
