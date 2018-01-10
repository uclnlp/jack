# -*- coding: utf-8 -*-

from jack.eval.link_prediction import compute_ranks

triple_to_score_map = {
    ('a', 'p', 'a'): 1,
    ('a', 'p', 'b'): 2,
    ('a', 'p', 'c'): 3,
    ('a', 'p', 'd'): 4
}

triples = sorted(triple for triple, _ in triple_to_score_map.items())
entity_set = {s for (s, _, _) in triples} | {o for (_, _, o) in triples}


def scoring_function(triples):
    return [triple_to_score_map.get(triple, 0) for triple in triples]


def test_kbp_eval():
    ranks, f_ranks = compute_ranks(scoring_function=scoring_function, triples=triples, entity_set=entity_set)

    ranks_l, ranks_r = ranks
    f_ranks_l, f_ranks_r = f_ranks

    assert ranks_l == [1, 1, 1, 1]
    assert ranks_r == [4, 3, 2, 1]

    assert f_ranks_l == ranks_l
    assert f_ranks_r == ranks_r
