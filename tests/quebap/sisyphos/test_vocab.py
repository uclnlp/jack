# -*- coding: utf-8 -*-

from quebap.sisyphos import vocab


def test_vocab():
    v = vocab.Vocab()
    assert v('A') == 1
    assert v('B') == 2
    assert v('C') == 3
    assert v('A') == 1
    assert v('B') == 2
