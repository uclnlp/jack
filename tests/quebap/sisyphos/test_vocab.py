# -*- coding: utf-8 -*-

from quebap.sisyphos import vocab


def test_vocab():
    v = vocab.Vocab()

    assert v('A') == 1
    assert v('B') == 2
    assert v('C') == 3
    assert v('A') == 1
    assert v('B') == 2
    assert v('D') == 4

    assert v.sym2id == {'<UNK>': 0, 'B': 2, 'D': 4, 'A': 1, 'C': 3}
    assert v.id2sym == {0: '<UNK>', 1: 'A', 2: 'B', 3: 'C', 4: 'D'}

    assert v.get_ids_oov() == [0, 1, 2, 3, 4]

    v.freeze()
    assert v('E') == 0
    v.unfreeze()
    assert v('E') == 5

    assert v.get_ids_pretrained() == []
    assert v.get_ids_oov() == [0, 1, 2, 3, 4, 5]
