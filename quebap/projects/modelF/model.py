import tensorflow as tf
from tensorflow import ConfigProto

import numpy as np
from numpy.random import randint
from util import *
from copy import copy
from time import time

SEED = 42

def define_model_F_score(tuples, entity_pair_embeddings, relation_embeddings):
    '''
    Returns dot products of relation embedding and entity pair embedding,
    batchwise. 'tuples' is a matrix: [batchsize x 2].
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


def nll_loss(dot_products, gold_labels, normalize=True):
    '''
    Given a set of dot product scores for positive facts [batchsize x 1]
    and a set of dot product scores for negative facts [batchsize x 1]
    compute negative log-likelihood loss and return single scalar value.
    Normalization divides loss by batchsize.
    '''
    # filter the dot products of true facts from those of false facts
    dot_products_false, dot_products_true = \
                            tf.dynamic_partition(dot_products, gold_labels, 2)

    # compute nll loss contribution for each part
    true_contribution = tf.nn.softplus(-dot_products_true)
    false_contribution = tf.nn.softplus(dot_products_false)
    if normalize:   # batch-normalization
        total_batch_loss = tf.reduce_mean(true_contribution \
                                            + false_contribution)/2.0
    else:
        total_batch_loss = tf.reduce_sum(true_contribution + false_contribution)
    return total_batch_loss


def batcher(batchsize, training_data, negative_domain_size):
    '''
    Create batch of training instances, half true facts, half false facts.
    '''
    ### create positive training instances
    n = training_data.shape[0]
    random_example_indices = randint(low=0, high=n, size=batchsize)
    true_training_instances = training_data[random_example_indices,:]

    ### create negative training instances
    # pick random indices (of entity pairs)
    random_negative_indices = randint(0, negative_domain_size,
                                                size=batchsize)
    # copy the true training instances,
    false_training_instances = copy(true_training_instances)
    # but replace the entity pair entry with random values.
    false_training_instances[:,1] = random_negative_indices

    ### gold values (True/ False)
    gold_true = np.ones([batchsize], dtype=int)
    gold_false = np.zeros([batchsize], dtype=int)

    ### concatenate and shuffle order randomly
    training_facts = np.concatenate((true_training_instances,
                                         false_training_instances))
    gold = np.concatenate(( gold_true, gold_false))
    perm = np.random.permutation(batchsize*2)
    facts_shuffled = training_facts[perm, :]
    gold_shuffled = gold[perm]
    return facts_shuffled, gold_shuffled


def initialize_embeddings(n_entity_pairs, n_relations, k, sigma=1.0):
    '''
    Initialise Gaussian embeddings for entity pairs and relations with sigma
    the standard deviation and k the latent dimensionality.
    '''
    entity_pair_embeddings = tf.Variable(tf.random_normal([n_entity_pairs, k], \
                                         0.0, sigma, seed=randint(SEED)))

    relation_embeddings = tf.Variable(tf.random_normal([n_relations, k], \
                                      0.0, sigma, seed=randint(SEED)))
    return entity_pair_embeddings, relation_embeddings


def training(loss, batch_function, placeholders, data, batchsize=32,
             learning_rate=0.1, optimization='Adam', n_iterations=20):
    ''' Training procedure for optimising a predefined loss over data using a
    batch_function to create training instances that are fed into the input
    placeholders.
    '''
    n_entity_pairs = 3
    # define optimizer
    if optimization == 'SGD':
        print("using SGD trainer")
        SGD = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        update_op = SGD.minimize(tf.reduce_mean(loss))
    else:
        print("using Adam trainer")
        Adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_op = Adam.minimize(tf.reduce_mean(loss))

    # Pontus' advice on efficiency in parallel...
    config = ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
            )
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    t0 = time()
    for iteration in range(0, n_iterations):
        minibatch_facts, minibatch_gold = batch_function(batchsize, data)
        feed = {placeholders["facts"]: minibatch_facts,
                placeholders["gold"]: minibatch_gold
                }
        _, loss__ = sess.run((update_op, loss), feed_dict=feed)
        print(iteration, loss__)
    return sess



def test():
    k = 5
    batchsize = 3
    n_data = 7
    n_entity_pairs = 13
    n_relations = 17
    E2_emb, R_emb = initialize_embeddings(n_entity_pairs, n_relations, k)

    placeholders = {"facts": tf.placeholder(dtype=tf.int64, \
                                                shape=(None,2), name='facts'),
                    "gold": tf.placeholder(dtype=tf.int64, \
                                                shape=None, name='gold')
                    }
    data = randint(0, n_data, [10, 2])
    facts, gold = batcher(batchsize, data, 10)
    print('data', data)
    print('true and false facts shuffled', facts)
    print('gold shuffled', gold)

    dot_product_scores = define_model_F_score(facts, E2_emb, R_emb)
    loss = nll_loss(dot_product_scores, gold, normalize=True)
    # specify number of entity pairs for batcher
    batch_function = (lambda bs, d: batcher(bs, d, n_entity_pairs))
    training(loss, batch_function, placeholders, data)


if __name__ == "__main__":
    test()
