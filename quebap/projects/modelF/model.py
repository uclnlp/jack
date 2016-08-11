import tensorflow as tf
import numpy as np
from util import *


def define_model_F_score(tuples, entity_pair_embeddings, relation_embeddings):
    '''
    Returns dot products of relation embedding and ent. pair embedding, batchwise.
    tuples is a matrix  [batchsize x 2].
    Each row contains an integer index for the relation, one for entity pair.
    entity_pair_embeddings: [N_entity_pairs x k]
    relation_embeddings: [N_relations x k]

    '''
    # each [batchsize x 1]
    relation_indices, ent_pair_indices = tf.split (1, 2, tuples)

    # each [batchsize x 1 x k]
    selected_embeddings_rel = tf.gather(relation_embeddings, relation_indices)
    selected_embeddings_e2 = tf.gather(entity_pair_embeddings, ent_pair_indices)

    # [batchsize x 1 x k]
    multiplied = tf.mul(selected_embeddings_rel, selected_embeddings_e2)

    # [batchsize x 1]
    dot_products = tf.reduce_sum(multiplied, 2)
    return dot_products


def test():
    k = 5
    batchsize = 17
    n = 10
    tuples = tf.Variable(np.random.randint(0, n, [batchsize, 2]))
    E2_emb = tf.Variable(np.random.normal(0.0, 1.0, [n, k]))
    R_emb = tf.Variable(np.random.normal(0.0, 1.0, [n, k]))
    dot_products = define_model_F_score(tuples, E2_emb, R_emb)
    tfrunprint(dot_products)


if __name__ == "__main__":
    test()
