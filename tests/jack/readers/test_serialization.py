# -*- coding: utf-8 -*-

from jack.readers.implementations import *
from jack.io.load import loaders

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

        if data is not None:
            shared_resources = SharedResources(vocab, config)
            reader_instance = reader(shared_resources)
            reader_instance.setup_from_data(data)

            temp_dir_path = tempfile.mkdtemp()
            reader_instance.store(temp_dir_path)

            reader.load(temp_dir_path)


test_serialization()
