# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

#from quebap.sisyphos import scratch


def test_get_embedder():
    input_size = 10
    vocab_size = 1000

    seq = tf.placeholder(tf.int64, [None, None], 'seq')
    #seq_embedded = scratch.embedder(seq, input_size, vocab_size)

