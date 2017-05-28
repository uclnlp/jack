# -*- coding: utf-8 -*-

import numpy as np


def compute_ranks(scoring_function, triples, entity_set, true_triples=None):
    subject_ranks, object_ranks = [], []
    subject_ranks_filtered, object_ranks_filtered = [], []

    for s, p, o in triples:
        subject_triples = [(s, p, o)] + [(x, p, o) for x in entity_set if x != s]
        object_triples = [(s, p, o)] + [(s, p, x) for x in entity_set if x != o]

        subject_triple_scores = np.array(scoring_function(subject_triples))
        object_triple_scores = np.array(scoring_function(object_triples))

        subject_rank = 1 + np.argsort(np.argsort(- subject_triple_scores))[0]
        object_rank = 1 + np.argsort(np.argsort(- object_triple_scores))[0]

        subject_ranks.append(subject_rank)
        object_ranks.append(object_rank)

        if true_triples:
            for idx, triple in enumerate(subject_triples):
                if triple != (s, p, o) and triple in true_triples:
                    subject_triple_scores[idx] = - np.inf

            for idx, triple in enumerate(object_triples):
                if triple != (s, p, o) and triple in true_triples:
                    object_triple_scores[idx] = - np.inf

            subject_rank_filtered = 1 + np.argsort(np.argsort(- subject_triple_scores))[0]
            object_rank_filtered = 1 + np.argsort(np.argsort(- object_triple_scores))[0]

            subject_ranks_filtered.append(subject_rank_filtered)
            object_ranks_filtered.append(object_rank_filtered)

    return (subject_ranks, object_ranks), (subject_ranks_filtered, object_ranks_filtered)

if __name__ == '__main__':
    pass
