import tensorflow as tf
import numpy as np
from util import *
from copy import copy


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
    random_example_indices = np.random.randint(low=0, high=n, size=batchsize)
    true_training_instances = training_data[random_example_indices,:]

    ### create negative training instances
    # pick random indices (of entity pairs)
    random_negative_indices = np.random.randint(0, negative_domain_size,
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


def test():
    k = 5
    batchsize = 3
    n = 10
    E2_emb = tf.Variable(np.random.normal(0.0, 1.0, [n, k]))
    R_emb = tf.Variable(np.random.normal(0.0, 1.0, [n, k]))
    data = np.random.randint(0, n, [10, 2])
    facts, gold = batcher(batchsize, data, 10)
    print('data', data)
    print('true and false facts shuffled', facts)
    print('gold shuffled', gold)
    dot_product_scores = define_model_F_score(facts, E2_emb, R_emb)
    loss = nll_loss(dot_product_scores, gold, normalize=True)
    tfrunprint(loss)



if __name__ == "__main__":
    test()
