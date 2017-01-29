from pytest import xfail

import jtr.jack as jack
import jtr.example as example
from jtr.data_structures import *
from jtr.preprocess.vocab import Vocab


@xfail("Not implemented yet")
def test_example_reader_overfit():
    vocab = jack.SharedVocab(Vocab())
    input_module = example.ExampleInputModule(vocab)
    model_module = example.ExampleModelModule(vocab)
    output_module = example.ExampleOutputModule()
    reader = jack.Reader(input_module, model_module, output_module, vocab)

    example_input = Input(["the father of Bart is Homer", "the father of Homer is Abe"], "Who is the father of Homer?",
                          ["Homer", "Bart", "Abe", "Lisa"])
    train_data = [(example_input, "Abe")]

    dev_data = train_data
    test_data = train_data

    reader.train(train_data, dev_data, test_data)

    answer = reader([example_input])

    assert answer[0].text == "Abe"
