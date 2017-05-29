# -*- coding: utf-8 -*-
from jtr.core import SharedResources

from jtr.util.vocab import Vocab
from jtr.util.global_config import Config

def test_SharedResources():
    shared_resources = SharedResources()
    assert shared_resources.vocab is not None
    assert shared_resources.config is not None
    assert type(shared_resources.vocab) == Vocab
    assert type(shared_resources.config) == Config
