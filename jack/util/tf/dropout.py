# -*- coding: utf-8 -*-

import tensorflow as tf


def fixed_dropout(xs, keep_prob, noise_shape, seed=None):
    """
    Apply dropout with same mask over all inputs
    Args:
        xs: list of tensors
        keep_prob:
        noise_shape:
        seed:

    Returns:
        list of dropped inputs
    """
    with tf.name_scope("dropout", values=xs):
        noise_shape = noise_shape
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape, seed=seed, dtype=xs[0].dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)
        outputs = []
        for x in xs:
            ret = tf.div(x, keep_prob) * binary_tensor
            ret.set_shape(x.get_shape())
            outputs.append(ret)
        return outputs
