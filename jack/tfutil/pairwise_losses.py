# -*- coding: utf-8 -*-

import sys

import tensorflow as tf


def hinge_loss(positive_scores, negative_scores, margin=1.0):
    """
    Pairwise hinge loss [1]:
        loss(p, n) = \sum_i [\gamma - p_i + n_i]_+

    [1] http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf

    Args:
        positive_scores: (N,) Tensor containing scores of positive examples.
        negative_scores: (N,) Tensor containing scores of negative examples.
        margin: Margin.
    Returns:
        Loss value.
    """
    hinge_losses = tf.nn.relu(margin - positive_scores + negative_scores)
    loss = tf.reduce_sum(hinge_losses)
    return loss


def logistic_loss(positive_scores, negative_scores):
    """
    Pairwise logistic loss [1]:
        loss(p, n) = \sum_i log(1 + e^(1 - p_i + n_i))

    [1] http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf

    Args:
        positive_scores: (N,) Tensor containing scores of positive examples.
        negative_scores: (N,) Tensor containing scores of negative examples.
    Returns:
        Loss value.
    """
    logistic_losses = tf.log(1 + tf.exp(1 - positive_scores + negative_scores))
    loss = tf.reduce_sum(logistic_losses)
    return loss


def mce_loss(positive_scores, negative_scores):
    """
    Minimum Classification Error (MCE) loss [1]:
        loss(p, n) = \sum_i \sigma(- p_i + n_i)

    [1] http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf

    Args:
        positive_scores: (N,) Tensor containing scores of positive examples.
        negative_scores: (N,) Tensor containing scores of negative examples.
    Returns:
        Loss value.
    """
    mce_losses = tf.sigmoid(- positive_scores + negative_scores)
    loss = tf.reduce_sum(mce_losses)
    return loss


def square_square_loss(positive_scores, negative_scores, margin=1.0):
    """
    Square-Square loss [1]:
        loss(p, n) = \sum_i - p_i^2 + [\gamma + n_i]^2_+

    [1] http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf

    Args:
        positive_scores: (N,) Tensor containing scores of positive examples.
        negative_scores: (N,) Tensor containing scores of negative examples.
    :   margin: Margin.
    Returns:
        Loss value.
    """
    square_square_losses = - positive_scores + tf.nn.relu(margin + negative_scores) ** 2
    loss = tf.reduce_sum(square_square_losses)
    return loss


def square_exponential_loss(positive_scores, negative_scores, gamma=1.0):
    """
    Square-Exponential loss [1]:
        loss(p, n) = \sum_i - p_i^2 + \gamma e^(n_i)

    [1] http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf

    Args:
        positive_scores: (N,) Tensor containing scores of positive examples.
        negative_scores: (N,) Tensor containing scores of negative examples.
        gamma: Gamma hyper-parameter.
    Returns:
        Loss value.
    """
    square_exponential_losses = - positive_scores + gamma * tf.exp(negative_scores)
    loss = tf.reduce_sum(square_exponential_losses)
    return loss


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown objective function: {}'.format(function_name))
    return getattr(this_module, function_name)

