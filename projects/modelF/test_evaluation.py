# -*- coding: utf-8 -*-

from . evaluation import RR


def test_reciprocal_rank():
    assert RR(4) == 0.25
