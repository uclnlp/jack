# -*- coding: utf-8 -*-

import tensorflow as tf


def mask_for_lengths(lengths, max_length=None, mask_right=True, value=-1000.0):
    """
    Creates a [batch_size x max_length] mask.

    Args:
        lengths: int32 1-dim tensor of batch_size lengths
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
