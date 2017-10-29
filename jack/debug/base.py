# -*- coding: utf-8 -*-

import tensorflow as tf

import logging

logger = logging.getLogger(__name__)


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
