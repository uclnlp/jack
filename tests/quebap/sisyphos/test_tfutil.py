# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from quebap.sisyphos import tfutil


def test_unit_length_transform():
    var = tf.Variable(tf.random_normal([32, 10], stddev=0.01), name='var')
    uvar = tfutil.unit_length_transform(var)

    init_op = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init_op)
        uvar_value = session.run(uvar)

        np.testing.assert_almost_equal(np.linalg.norm(uvar_value, ord=2, axis=1), 1)
