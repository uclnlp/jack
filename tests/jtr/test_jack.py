# -*- coding: utf-8 -*-

import pytest

import jtr.core as core
import jtr.tasks.mcqa.simple_mcqa as example
from jtr.data_structures import *
from jtr.util.vocab import Vocab


@pytest.mark.skip("Not implemented yet")
def test_example_reader_overfit():
    vocab = core.SharedVocabAndConfig(Vocab())
    input_module = example.SimpleMCInputModule(vocab)
    model_module = example.SimpleMCModelModule(vocab)
    output_module = example.SimpleMCOutputModule()
    reader = core.JTReader(vocab, input_module, model_module, output_module)

    example_input = QASetting("Who is the father of Homer?",
                              ["the father of Bart is Homer", "the father of Homer is Abe"],
                              atomic_candidates=["Homer", "Bart", "Abe", "Lisa"])
    train_data = [(example_input, Answer("Abe"))]

    dev_data = train_data
    test_data = train_data

    reader.train(train_data, dev_data, test_data)

    answer = reader([example_input])

    assert answer[0].text == "Abe"
