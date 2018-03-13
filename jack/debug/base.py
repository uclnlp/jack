# -*- coding: utf-8 -*-

import tensorflow as tf

import logging

logger = logging.getLogger(__name__)


def get_total_trainable_variables():
    """Calculates and returns the number of trainable parameters in the model."""
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def get_total_variables():
    """Calculates and returns the number of parameters in the model (these can be fixed)."""
    total_parameters = 0
    for variable in tf.global_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def test_update(feed_dict, train_op):
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    before = session.run(tf.trainable_variables())
    session.run(train_op, feed_dict=feed_dict)
    after = session.run(tf.trainable_variables())

    res = False
    for b, a in zip(before, after):
        # Check if anything changed
        res |= (b != a).any()

    return res


def test_loss(feed_dict, loss_op):
    session = tf.Session()
    loss = session.run(loss_op, feed_dict=feed_dict)
    return loss != 0
