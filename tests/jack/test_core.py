# -*- coding: utf-8 -*-

import numpy as np

from jack.core import SharedResources
from jack.io.embeddings import load_embeddings
from jack.util.vocab import Vocab


def test_shared_resources_store():
    embeddings_file = "data/GloVe/glove.the.50d.txt"
    embeddings = load_embeddings(embeddings_file, 'glove')
    config = {
        "embedding_file": embeddings_file,
        "embedding_format": "glove"
    }
    some_vocab = Vocab(vocab=embeddings.vocabulary)
    some_vocab('foo')
    shared_resources = SharedResources(some_vocab, config, embeddings)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = tmp_dir + "_resources"
        shared_resources.store(path)

        new_shared_resources = SharedResources()
        new_shared_resources.load(path)

        type_a, type_b = type(new_shared_resources.vocab), type(shared_resources.vocab)
        assert type_a == type_b

        for k in new_shared_resources.vocab.__dict__:
            assert new_shared_resources.vocab.__dict__[k] == shared_resources.vocab.__dict__[k]
        assert new_shared_resources.config == shared_resources.config
        assert new_shared_resources.embeddings.lookup.shape == embeddings.lookup.shape
        assert np.array_equal(new_shared_resources.embeddings.get(b"the"), embeddings.get(b"the"))
