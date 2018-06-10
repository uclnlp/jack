# -*- coding: utf-8 -*-

import tensorflow as tf


def parametric_relu(x, name=None):
    alphas = tf.get_variable('{}/alpha'.format(name) if name else 'alpha',
                             x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    return tf.nn.relu(x) + alphas * (x - abs(x)) * 0.5


def selu(x, name=None):
    with tf.name_scope('{}/elu'.format(name) if name else 'elu') as _:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))


# Aliases
prelu = parametric_relu


def activation_from_string(activation_str):
    if activation_str is None:
        return tf.identity
    return getattr(tf.nn, activation_str)
