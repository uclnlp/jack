# -*- coding: utf-8 -*-

from jack.readers.implementations import *
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

        shared_resources = SharedResources(vocab, config)
        reader_instance = reader(shared_resources)

        temp_dir_path = tempfile.mkdtemp()
        reader_instance.store(temp_dir_path)

        reader.load(temp_dir_path)


test_serialization()
