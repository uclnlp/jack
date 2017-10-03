# -*- coding: utf-8 -*-

import tensorflow as tf


def parametric_relu(x, name=None):
    alphas = tf.get_variable('{}/alpha'.format(name) if name else 'alpha',
                             x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    return tf.nn.relu(x) + alphas * (x - abs(x)) * 0.5

# Aliases
prelu = parametric_relu
