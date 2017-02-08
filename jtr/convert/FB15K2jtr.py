"""

jtr converter for the fb15k dataset.

Bordes, Antoine, et al.
"Translating embeddings for modeling multi-relational data."
Advances in neural information processing systems. 2013.

Original paper:
        https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
Data:   https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz
Web:    https://everest.hds.utc.fr/doku.php?id=en:transe
JTR download script: data/FB15k/download.sh

Metadata:

Training data:
    483142 triples (subject, relation, object)
    14951 different entities
    1345 different relation types

"""

from collections import defaultdict
import gc
import json
import argparse


def load_fb15k_triples(path):
    """
    Loads the raw data from file provided.

    Args:
        path: path to the file

    Returns: triples
    """
    with open(path, 'r') as f:
        triples = [line.strip('\n').split('\t') for line in f.readlines()]
    return triples


def extract_unique_entities_and_relations(triples):
    """
    Identifies unique entities and relation types in collection of triples.

    Args:
        triples: List of string triples.

    Returns:
        unique_entities: List of strings
        unique_relations: List of strings
    """
    s_entities = set([triple[0] for triple in triples])
    o_entities = set([triple[2] for triple in triples])
    r_types = set([triple[1] for triple in triples])

    unique_relations = sorted(list(r_types))
    unique_entities = sorted(list(s_entities | o_entities))  # union of sets

    return unique_entities, unique_relations


def get_facts_per_entity(triples):
    """
    Obtain dictionary with all train fact ids that contain an entity.

    Args:
        triples: List of fact triples

    Returns:
        Dictionary entity --> fact IDs it participates in
    """
    d = defaultdict(set)
    for i_triple, triple in enumerate(triples):
        d[triple[0]].add(i_triple)
        d[triple[2]].add(i_triple)
    return d


def get_facts_per_relation(triples):
    """
    Obtain dictionary with all train fact ids that contain a relation type.

    Args:
        triples: List of fact triples

    Returns:
        Dictionary relation type --> fact IDs it participates in
    """
    d = defaultdict(set)
    for i_triple, triple in enumerate(triples):
        d[triple[1]].add(i_triple)
    return d


def get_fact_neighbourhoods(triples, facts_per_entity, facts_per_relation,
                            include_relations=False):
    """
    Extracts neighbouring facts for a collection of triples. neighbouring
    facts of fact f are such facts that share at least an entity with f.
    If relations are included, facts which share a relation are also considered
    neighbours.

    Args:
        triples: list of facts triples
        facts_per_entity: dictionary; The facts an entity appears in
        facts_per_relation: dictionary; The facts a relation appears in
        include_relations: boolean. whether facts sharing the relation should
            be considered neighbours as well.

    Returns:
        fact_neighbourhoods: dictionary mapping fact ID to set of fact IDs.
    """
    fact_neighbourhoods = defaultdict(set)
    for i_triple, triple in enumerate(triples):
        # get triple ids which share subject, object or rel. with current triple
        subject_neighbours = facts_per_entity[triple[0]]
        object_neighbours = facts_per_entity[triple[2]]
        relation_neighbours = set()
        if include_relations:
            relation_neighbours = facts_per_relation[triple[1]]

        fact_neighbourhoods[i_triple].update(subject_neighbours)
        fact_neighbourhoods[i_triple].update(object_neighbours)
        fact_neighbourhoods[i_triple].update(relation_neighbours)

    return fact_neighbourhoods


def convert_fb15k2(triples, neighbourhoods, unique_entities):
    """
    Converts into jtr format.
    Args:
        triples: fact triples that should be converted.
        neighbourhoods: dictionary of supporting facts per triple
        unique_entities: List of strings

    Returns:
        jtr formatted fb15k data.
    """
    # figure out cases with multiple possible true answers
    multiple_answers_dict = defaultdict(set)
    for triple in triples:
        multiple_answers_dict[triple[:2]].add(triple[2])

    instances = []
    for i, triple in enumerate(triples):
        if not i % 1000:
            # print(i)
            gc.collect()
        # correct answers for this (s,r,.) case
        correct_answers = multiple_answers_dict[triple[:2]]

        # obtain supporting facts for this triple
        neighbour_ids = neighbourhoods[i]
        neighbour_triples = [triples[ID] for ID in neighbour_ids]

        # create a single jtr instance
        qset_dict = {}
        support_texts = [" ".join([str(s), str(r), str(o)]) for (s, r, o) in neighbour_triples]

        qset_dict['support'] = [{'text': t} for t in support_texts]
        qset_dict['questions'] = [{
            "question": " ".join([str(triple[0]), str(triple[1])]),  # subject and relation
            "candidates": [],  # use global candidates instead.
            "answers": [{'text': str(a)} for a in correct_answers]  # object
        }]
        instances.append(qset_dict)

    return {
        'meta': 'FB15K with entity neighbours as supporting facts.',
        'globals': {
            'candidates': [{'text': str(i)} for (i, u) in enumerate(unique_entities)]
        },
        'instances': instances
    }


def compress_triples(string_triples, unique_entities, unique_relations):
    id_triples = []
    dict_unique_entities = {elem: i for i, elem in enumerate(unique_entities)}
    dict_unique_relations = {elem: i for i, elem in enumerate(unique_relations)}
    for (s, r, o) in string_triples:
        s_id = dict_unique_entities[s]
        r_id = dict_unique_relations[r]
        o_id = dict_unique_entities[o]
        id_triples.append((s_id, r_id, o_id))
    return id_triples


def main():
    parser = argparse.ArgumentParser(description='FB15K2 dataset to jtr format converter.')
    #
    parser.add_argument('infile',
                        help="dataset path you're interested in, train/dev/test."
                             "(e.g. data/FB15k/FB15k/freebase_mtr100_mte100-train.txt)")
    parser.add_argument('reffile',
                        help="reference file - use training set path here.")
    parser.add_argument('outfile',
                        help="path to the jtr format -generated output file (e.g. data/FB15K2/FB15k_train.jtr.json)")
    # parser.add_argument('dataset', choices=['cnn', 'dailymail'],
    #                     help="which dataset to access: cnn or dailymail")
    # parser.add_argument('split', choices=['train', 'dev', 'test'],
    #                     help="which split of the dataset to convert: train, dev or test")
    args = parser.parse_args()

    # load data from files into fact triples
    triples = load_fb15k_triples(args.infile)
    reference_triples = load_fb15k_triples(args.reffile)

    # unique entity and relation types in reference triples
    unique_entities, unique_relations = \
        extract_unique_entities_and_relations(reference_triples)
    # represent string triples with numeric IDs for entities and relations
    triples = compress_triples(triples, unique_entities, unique_relations)
    reference_triples = compress_triples(reference_triples, unique_entities, unique_relations)

    # get neighbouring facts for each fact in triples
    facts_per_entity = get_facts_per_entity(reference_triples)
    facts_per_relation = get_facts_per_relation(reference_triples)
    neighbourhoods = get_fact_neighbourhoods(triples, facts_per_entity, facts_per_relation)

    # dump the entity and relation ids for understanding the jtr contents.
    with open('fb15k_entities_relations.json', 'w') as f:
        d = {"unique_entities": unique_entities,
             "unique_relations": unique_relations}
        json.dump(d, f)

    corpus = convert_fb15k2(triples, neighbourhoods, unique_entities)
    with open(args.outfile, 'w') as outfile:
        json.dump(corpus, outfile, indent=2)


if __name__ == "__main__":
    main()
