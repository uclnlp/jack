# -*- coding: utf-8 -*-

import sys

import tensorflow as tf


def negative_l1_distance(x1, x2, axis=1):
    """
    Negative L1 Distance.

    .. math:: L = - \\sum_i \\abs(x1_i - x2_i)

    Args:
        x1: First term.
        x2: Second term.
        axis: Reduction Indices.

    Returns:
        Similarity Value.
    """
    distance = tf.reduce_sum(tf.abs(x1 - x2), axis=axis)
    return - distance


def negative_l2_distance(x1, x2, axis=1):
    """
    Negative L2 Distance.

    .. math:: L = - \\sqrt{\\sum_i (x1_i - x2_i)^2}

    Args:
        x1: First term.
        x2: Second term.
        axis: Reduction Indices.

    Returns:
        Similarity Value.
    """

    distance = tf.sqrt(tf.reduce_sum(tf.square(x1 - x2), axis=axis))
    return - distance


def negative_square_l2_distance(x1, x2, axis=1):
    """
    Negative Square L2 Distance.

    .. math:: L = - \\sum_i (x1_i - x2_i)^2

    Args:
        x1: First term.
        x2: Second term.
        axis: Reduction Indices.

    Returns:
        Similarity Value.
    """
    distance = tf.reduce_sum(tf.square(x1 - x2), axis=axis)
    return - distance


def dot_product(x1, x2, axis=1):
    """
    Dot Product.

    .. math:: L = \\sum_i x1_i x2_i

    Args:
        x1: First term.
        x2: Second term.
        axis: Reduction Indices.

    Returns:
        Similarity Value.
    """

    similarity = tf.reduce_sum(x1 * x2, axis=axis)
    return similarity


# Aliases
l1 = L1 = negative_l1_distance
l2 = L2 = negative_l2_distance
l2_sqr = L2_SQR = negative_square_l2_distance
dot = DOT = dot_product


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown similarity function: {}'.format(function_name))
    return getattr(this_module, function_name)
