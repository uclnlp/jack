# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from quebap.sisyphos import vocab


def test_vocab():
    v = vocab.Vocab()

    assert v('A') == 1
    assert v('B') == 2
    assert v('C') == 3
    assert v('A') == 1
    assert v('B') == 2
    assert v('D') == 4

    assert v.sym2id == {'<UNK>': 0, 'B': 2, 'D': 4, 'A': 1, 'C': 3}
    assert v.id2sym == {0: '<UNK>', 1: 'A', 2: 'B', 3: 'C', 4: 'D'}

    assert v.get_ids_oov() == [0, 1, 2, 3, 4]

    v.freeze()
    assert v('E') == 0
    v.unfreeze()
    assert v('E') == 5

    assert v.get_ids_pretrained() == []
    assert v.get_ids_oov() == [0, 1, 2, 3, 4, 5]


def test_neural_vocab():
    def emb(w):
        v = {'A': [1.7, 0, .3], 'B': [0, 1.5, 0.5], 'C': [0, 0, 2]}
        return v.get(w, None)

    v = vocab.Vocab(emb=emb)
    v('A', 'B', 'C', 'hello', 'world')
    v(['B', 'world', 'wake', 'up'])

    with tf.variable_scope('neural_test'):
        nv = vocab.NeuralVocab(v, None, 3, unit_normalize=False)

    init_op = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init_op)
        np.testing.assert_almost_equal(session.run(nv(v('A'))), [1.7, 0, .3])
        np.testing.assert_almost_equal(session.run(nv(v('B'))), [0., 1.5, 0.5])
        np.testing.assert_almost_equal(session.run(nv(v('C'))), [0., 0., 2.])

