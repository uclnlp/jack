# -*- coding: utf-8 -*-
from jtr.core import SharedResources
from jtr.util.vocab import Vocab



def test_SharedResources():
    shared_resources = SharedResources()

    some_vocab = Vocab(emb=None)
    some_vocab('someword')
    shared_resources.answer_vocab = some_vocab

    shared_resources.store('tmp/somedummy.pickle')
    loaded_shared_resources = shared_resources.load('tmp/somedummy.pickle')

    assert loaded_shared_resources.answer_vocab == shared_resources.answer_vocab


test_SharedResources()
