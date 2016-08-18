from . evaluation import RR


def test_reciprocal_rank():
    assert RR(4) == 0.25


def test_reciprocal_rank_2():
    assert RR(2) == 0.53
