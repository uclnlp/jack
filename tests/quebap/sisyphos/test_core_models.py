# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from quebap.sisyphos.core_models import ConditionalLSTMCell


def test_conditional_lstm_cell():
    input_size = 2
    output_size = 3
    batch_size = 5
    max_length = 7

    cell = ConditionalLSTMCell(output_size)

    input_embedded = tf.placeholder(tf.float32, [None, None, input_size], "input_embedded")
    input_length = tf.placeholder(tf.int64, [None], "input_length")
    outputs, states = tf.nn.dynamic_rnn(cell, input_embedded, sequence_length=input_length, dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        feed_dict = {
            input_embedded: np.random.randn(batch_size, max_length, input_size),
            input_length: np.random.randint(1, max_length, batch_size)
        }

        states_value = sess.run(states, feed_dict).h
        assert states_value.shape == (5, 3)
