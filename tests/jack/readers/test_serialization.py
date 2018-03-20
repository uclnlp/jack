# -*- coding: utf-8 -*-

from jack.readers.implementations import *
from jack.io.load import loaders
from jack.io.embeddings import load_embeddings
from jack.util.vocab import Vocab

import tensorflow as tf

import tempfile


def test_serialization():
    all_readers = [
        fastqa_reader,
        modular_qa_reader,
        # fastqa_reader_torch,
        dam_snli_reader,
        cbilstm_nli_reader,
        modular_nli_reader,
        distmult_reader,
        complex_reader,
        transe_reader,
    ]

    for reader in all_readers:
        vocab, config = {}, {}

        data = None
        if reader in {distmult_reader, complex_reader, transe_reader}:
            data = loaders['jack']('tests/test_data/WN18/wn18-snippet.jack.json')
            config['repr_dim'] = 50
        elif reader in {cbilstm_nli_reader, dam_snli_reader}:
            data = loaders['snli']('tests/test_data/SNLI/1000_samples_snli_1.0_train.jsonl')

            embeddings = load_embeddings("data/GloVe/glove.the.50d.txt", 'glove')
            vocab = Vocab(emb=embeddings, init_from_embeddings=True)
            config['repr_dim_input'] = 50
            config['repr_dim'] = 50
        elif reader in {fastqa_reader}:
            data = loaders['squad']('data/SQuAD/snippet.json')

            embeddings = load_embeddings("data/GloVe/glove.the.50d.txt", 'glove')
            vocab = Vocab(emb=embeddings, init_from_embeddings=True)
            config['repr_dim_input'] = 50
            config['repr_dim'] = 50

        if data is not None:
            tf.reset_default_graph()

            shared_resources = SharedResources(vocab, config)
            reader_instance = reader(shared_resources)
            reader_instance.setup_from_data(data)

            temp_dir_path = tempfile.mkdtemp()
            reader_instance.store(temp_dir_path)

            reader_instance.load(temp_dir_path)

            assert reader_instance is not None


test_serialization()
