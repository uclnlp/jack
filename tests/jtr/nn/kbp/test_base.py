# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from jtr.nn.kbp.base import TranslatingModel, BilinearDiagonalModel, BilinearModel
from jtr.nn.kbp import similarities


def test_translating_embeddings_score():
    batch_size = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    E = rs.rand(batch_size, 2, embedding_size)
    R = rs.rand(batch_size, 1, embedding_size)

    vE = tf.Variable(E, name='E')
    vR = tf.Variable(R, name='R')

    model = TranslatingModel(subject_embeddings=vE[:, 0, :],
                             predicate_embeddings=vR[:, 0, :],
                             object_embeddings=vE[:, 1, :],
                             similarity_function=similarities.negative_l1_distance)
    scores = model()

    init_op = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init_op)

        scores_value = session.run(scores)
        assert(scores_value.shape[0] == batch_size)

        tmp = - np.sum(np.abs(E[:, 0, :] + R[:, 0, :] - E[:, 1, :]), axis=1)
        assert(np.isclose(scores_value, tmp).all())


def test_bilinear_diagonal_score():
    batch_size = 5
    embedding_size = 10

    rs = np.random.RandomState(0)

    E = rs.rand(batch_size, 2, embedding_size)
    R = rs.rand(batch_size, 1, embedding_size)

    vE = tf.Variable(E, name='E')
    vR = tf.Variable(R, name='R')

    model = BilinearDiagonalModel(subject_embeddings=vE[:, 0, :],
                                  predicate_embeddings=vR[:, 0, :],
                                  object_embeddings=vE[:, 1, :],
                                  similarity_function=similarities.negative_l1_distance)
    scores = model()

    init_op = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init_op)

        scores_value = session.run(scores)
        assert(scores_value.shape[0] == batch_size)

        tmp = - np.sum(np.abs(E[:, 0, :] * R[:, 0, :] - E[:, 1, :]), axis=1)
        assert(np.isclose(scores_value, tmp).all())


def test_bilinear_score():
    batch_size = 5
    entity_embedding_size = 2
    predicate_embedding_size = 4

    rs = np.random.RandomState(0)

    E = rs.rand(batch_size, 2, entity_embedding_size)
    R = rs.rand(batch_size, 1, predicate_embedding_size)

    vE = tf.Variable(E, name='E')
    vR = tf.Variable(R, name='R')

    model = BilinearModel(subject_embeddings=vE[:, 0, :],
                          predicate_embeddings=vR[:, 0, :],
                          object_embeddings=vE[:, 1, :],
                          similarity_function=similarities.dot_product)
    scores = model()

    init_op = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init_op)
        scores_value = session.run(scores)

        assert (scores_value.shape[0] == batch_size)

        for i in range(batch_size):
            es, eo = E[i, 0, :], E[i, 1, :]
            ep = np.reshape(R[i, 0, :], (entity_embedding_size, entity_embedding_size))
            np.testing.assert_allclose(scores_value[i], np.dot(np.dot(es, ep), eo))
