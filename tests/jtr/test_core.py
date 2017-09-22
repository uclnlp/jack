# -*- coding: utf-8 -*-
import tempfile

from jtr.core import SharedResources
from jtr.util.vocab import Vocab
from jtr.input_output.embeddings import load_embeddings
import numpy as np


def test_shared_resources():
    embeddings_file = "data/GloVe/glove.the.50d.txt"
    embeddings = load_embeddings(embeddings_file, 'glove')
    config = {
        "embedding_file": embeddings_file,
        "embedding_format": "glove"
    }
    some_vocab = Vocab()
    some_vocab('foo')
    shared_resources = SharedResources(some_vocab, config, embeddings)

    with tempfile.TemporaryDirectory() as tmp_dir:
        shared_resources.store(tmp_dir)

        new_shared_resources = SharedResources()
        new_shared_resources.load(tmp_dir)

        assert type(new_shared_resources.vocab) == type(shared_resources.vocab)
        assert new_shared_resources.vocab.__dict__ == shared_resources.vocab.__dict__
        assert new_shared_resources.config == shared_resources.config
        assert new_shared_resources.embeddings.lookup.shape == embeddings.lookup.shape
        assert np.array_equal(new_shared_resources.embeddings.get(b"the"), embeddings.get(b"the"))


def test_shared_resources_from_config():
    embeddings_file = "data/GloVe/glove.the.50d.txt"
    embeddings = load_embeddings(embeddings_file, 'glove')
    shared_resources = SharedResources.from_config(embedding_file=embeddings_file,
                                                   vocab_from_embeddings=True,
                                                   parent_config="conf/extractive_qa.yaml")
    from jtr.util.util import load_yaml_recursively
    parent_config = load_yaml_recursively("conf/extractive_qa.yaml")
    assert [k in shared_resources.config for k in parent_config.keys()]
    assert shared_resources.config['embedding_file'] == 'data/GloVe/glove.the.50d.txt'
    assert shared_resources.config['embedding_format'] == 'glove'
    assert shared_resources.config['vocab_from_embeddings'] is True
    assert shared_resources.vocab(Vocab.DEFAULT_UNK) == 1
    assert shared_resources.vocab(b"the") == 0
    assert len(shared_resources.vocab) == 2
    assert np.array_equal(shared_resources.embeddings.get(b"the"), embeddings.get(b"the"))
