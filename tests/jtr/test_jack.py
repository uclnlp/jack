from pytest import skip

import jtr.jack as jack
import jtr.jack.tasks.mcqa.simple_mcqa as example
from jtr.jack.data_structures import *
from jtr.preprocess.vocab import Vocab


@skip("Not implemented yet")
def test_example_reader_overfit():
    vocab = jack.SharedVocabAndConfig(Vocab())
    input_module = example.SimpleMCInputModule(vocab)
    model_module = example.SimpleMCModelModule(vocab)
    output_module = example.SimpleMCOutputModule()
    reader = jack.JTReader(input_module, model_module, output_module, vocab)

    example_input = Question(["the father of Bart is Homer", "the father of Homer is Abe"],
                             "Who is the father of Homer?",
                             ["Homer", "Bart", "Abe", "Lisa"])
    train_data = [(example_input, "Abe")]

    dev_data = train_data
    test_data = train_data

    reader.train(train_data, dev_data, test_data)

    answer = reader([example_input])

    assert answer[0].text == "Abe"
