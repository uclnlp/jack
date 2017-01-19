# -*- coding: utf-8 -*-

import tensorflow as tf


def negative_l1_distance(x1, x2, reduction_indices=1):
    """
    Negative L1 Distance.

    .. math:: L = - \\sum_i \\abs(x1_i - x2_i)

    :param x1: First term.
    :param x2: Second term.
    :param reduction_indices: Reduction Indices.
    :return: Similarity Value.
    """
    distance = tf.reduce_sum(tf.abs(x1 - x2), reduction_indices=reduction_indices)
    return - distance


def negative_l2_distance(x1, x2, reduction_indices=1):
    """
    Negative L2 Distance.

    .. math:: L = - \\sqrt{\\sum_i (x1_i - x2_i)^2}

    :param x1: First term.
    :param x2: Second term.
    :param reduction_indices: Reduction Indices.
    :return: Similarity Value.
    """

    distance = tf.sqrt(tf.reduce_sum(tf.square(x1 - x2), reduction_indices=reduction_indices))
    return - distance


def negative_square_l2_distance(x1, x2, reduction_indices=1):
    """
    Negative Square L2 Distance.

    .. math:: L = - \\sum_i (x1_i - x2_i)^2

    :param x1: First term.
    :param x2: Second term.
    :param reduction_indices: Reduction Indices.
    :return: Similarity Value.
    """
    distance = tf.reduce_sum(tf.square(x1 - x2), reduction_indices=reduction_indices)
    return - distance


def dot_product(x1, x2, reduction_indices=1):
    """
    Dot Product.

    .. math:: L = \\sum_i x1_i x2_i

    :param x1: First term.
    :param x2: Second term.
    :param reduction_indices: Reduction Indices.
    :return: Similarity Value.
    """

    similarity = tf.reduce_sum(x1 * x2, reduction_indices=reduction_indices)
    return similarity


# Aliases
l1 = L1 = negative_l1_distance
l2 = L2 = negative_l2_distance
l2_sqr = L2_SQR = negative_square_l2_distance
dot = DOT = dot_product
