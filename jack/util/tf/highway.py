# -*- coding: utf-8 -*-

import tensorflow as tf


def highway_layer(inputs, activation, name=None):
    with tf.variable_scope(name or "highway_layer"):
        d = inputs.get_shape()[-1].value
        trans_gate = tf.contrib.layers.fully_connected(inputs, 2 * d, activation_fn=None, weights_initializer=None,
                                                       scope='trans_gate')
        trans, gate = tf.split(trans_gate, 2, len(inputs.get_shape()) - 1)
        trans, gate = activation(trans), tf.sigmoid(gate)
        out = gate * trans + (1 - gate) * inputs
        return out


def highway_network(inputs, num_layers, activation=tf.tanh, name=None, reuse=False):
    with tf.variable_scope(name or "highway_network", reuse=reuse):
        prev = inputs
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, activation, name="layer_{}".format(layer_idx))
            prev = cur
    return cur
