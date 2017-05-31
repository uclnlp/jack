# -*- coding: utf-8 -*-
from jtr.core import SharedResources
from jtr.util.vocab import Vocab



def test_SharedResources():
    shared_resources = SharedResources()
    assert shared_resources

    some_vocab = Vocab()
    some_vocab('someword')
    shared_resources.vocab = some_vocab

    shared_resources.store('tmp/somedummy.pickle')

    new_shared_resources = SharedResources()
    new_shared_resources.load('tmp/somedummy.pickle')

    assert type(new_shared_resources.vocab) == type(shared_resources.vocab)
    assert new_shared_resources.vocab.__dict__ == shared_resources.vocab.__dict__
    assert new_shared_resources.config == shared_resources.config
