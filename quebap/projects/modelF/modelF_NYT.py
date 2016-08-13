import json
from structs import FrozenIdentifier

import numpy as np
import tensorflow as tf

from model import define_model_F_score, nll_loss
from model import initialize_embeddings, batcher, training


def load_train_data_from_quebap(filepath):
    '''
    returns matrix of training facts [n_instances x 2] where first column
    is for entity pair index and second is for relation type index.
    Indices are obtained using lexica for entity pairs and relations, which are
    also returned.
    '''
    with open(filepath, 'r') as f:
        quebap_file_contents = json.load(f)
    instances = quebap_file_contents['instances']

    # build lexica with unique entries for entity pairs and relations.
    all_entity_pair_candidates = set([c['text'] for c in \
                                quebap_file_contents['globals']['candidates'] ])

    all_relation_types = set([inst['questions'][0]['question'] for inst in instances])
    relation_lexicon = FrozenIdentifier(all_relation_types)
    entity_pair_lexicon = FrozenIdentifier(all_entity_pair_candidates)

    # convert instances into index format using lexica and fill into matrix.
    fact_matrix = np.zeros([len(instances), 2], dtype=int)
    for i, instance in enumerate(instances):
        rel_string = instance['questions'][0]['question']
        e2_string = instance['questions'][0]['answers'][0]['text']
        fact_matrix[i, 1] = entity_pair_lexicon._key2id[e2_string]
        fact_matrix[i, 0] = relation_lexicon._key2id[rel_string]

    gold = np.ones([len(instances),1], dtype=int)
    return fact_matrix, gold, entity_pair_lexicon, relation_lexicon


def load_test_data_from_from_quebap(filepath, entity_pair_lexicon,
                                    relation_lexicon):
    '''
    Read in NYT test data from quebap file, using filepath as location.
    entity_pair_lexicon and relation_lexicon are lexica for giving indices to
    the relation type/ entity pair strings and having this consistent with the
    training set. Outputs:
        - list of test entity pairs
    '''
    with open(filepath, 'r') as f:
        quebap_file_contents = json.load(f)

    ### get (negative) entity pair candidates to use during testing
    global_test_candidates = quebap_file_contents['globals']['candidates']
    preliminary_test_entity_pairs = [c['text'] for c in global_test_candidates]

    # Filter out entity pairs that didn't appear in the training set,
    # i.e. that are not in the lexicon.
    known_entity_pairs = set([x[0] for x in entity_pair_lexicon.items()])
    test_entity_pairs = list(known_entity_pairs.intersection \
                                        (set(preliminary_test_entity_pairs)))

    # go over all instances, extract relation and true entity pair strings,
    # convert into indices.
    relation_indices, e2_indices = [], []
    for instance in quebap_file_contents['instances']:
        rel_string = instance['questions'][0]['question']
        answer_strings = [a['text'] for a in instance['questions'][0]['answers']]
        relation_indices.append( relation_lexicon._key2id[rel_string] )
        e2_indices.append([entity_pair_lexicon._key2id[a] for a in answer_strings if a in known_entity_pairs])

    # convert test entity pairs into indices
    test_entity_pairs = [entity_pair_lexicon._key2id[ep] for ep in test_entity_pairs]

    # filter out relations with an answer entity pair not in known_entity_pairs
    for i, (r,e2) in enumerate(zip(relation_indices, e2_indices)):
        if len(e2) == 0:
            relation_indices.remove(r)
            e2_indices.remove(e2)
            
    # 25 relation types left overall.
    return test_entity_pairs, relation_indices, e2_indices


def test_procedure():
    pass



def main():
    import sys
    # load training data
    train_facts, gold, e_lexicon, r_lexicon = \
                    load_train_data_from_quebap('naacl2013_train.quebap.json')

    load_test_data_from_from_quebap('naacl2013_test.quebap.json', e_lexicon,
                                        r_lexicon)
    return -1

    # initialize embeddings
    n_entity_pairs = len(e_lexicon)
    n_relations = len(r_lexicon)
    k = 20
    E2_emb, R_emb = initialize_embeddings(n_entity_pairs, n_relations, k)

    # define input/output placeholders
    placeholders = {"facts": tf.placeholder(dtype=tf.int32, \
                                                shape=(None,2), name='facts'),
                    "gold": tf.placeholder(dtype=tf.int32, \
                                                shape=None, name='gold')
                    }

    # define model and loss
    dot_product_scores = define_model_F_score(placeholders['facts'], E2_emb, R_emb)
    loss = nll_loss(dot_product_scores, placeholders['gold'], normalize=True)

    # specify number of entity pairs for batcher
    batch_function = (lambda bs, d: batcher(bs, d, n_entity_pairs))
    training(loss, batch_function, placeholders, train_facts, batchsize=8192, n_iterations=2000)

    return -1
    if len(sys.argv) == 3:
        train_quebaps = withload_naacl2013(sys.argv[1])
        test_quebaps = sys.argv[2]
        print(json.dumps(data, indent=2))
    else:
        print("""usage: python3 naacl2013_train.quebap.json naacl2013_test.quebap.json > predictions.json """)

if __name__ == "__main__":
    main()
