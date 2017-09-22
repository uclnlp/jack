# -*- coding: utf-8 -*-
import tempfile

from jtr.core import SharedResources
from jtr.util.vocab import Vocab


def test_shared_resources():
    embeddings_file = "data/GloVe/glove.the.50d.txt"
    from jtr.input_output.embeddings import load_embeddings
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
        assert new_shared_resources.embeddings.get("the") == embeddings.get("the")
