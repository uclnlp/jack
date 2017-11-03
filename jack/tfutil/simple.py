# -*- coding: utf-8 -*-

import tensorflow as tf


def fully_connected_projection(inputs, output_size):
    """Projects inputs onto target dimension. Returns logits, loss, and argmax.

    Creates fully connected projection layer. Then applies cross entropy
    softmax to get the loss. Calculate predictions via argmax.
    Args:
        inputs (tensor): Input into the projection layer.
        output_size (int): Size of the targets (used in projection layer).
    """
    init = tf.contrib.layers.xavier_initializer(uniform=True) #uniform=False for truncated normal
    logits = tf.contrib.layers.fully_connected(inputs, output_size, weights_initializer=init, activation_fn=None)
    return logits
