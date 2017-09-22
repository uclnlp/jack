# -*- coding: utf-8 -*-

import pytest
import tensorflow as tf

from jtr.tasks.mcqa.simple_mcqa import *
from jtr.util.vocab import Vocab


@pytest.mark.skip("Not implemented yet")
def test_simple_mcqa():
    data_set = [
        (QASetting("which is it?", ["a is true", "b isn't"], atomic_candidates=["a", "b", "c"]),
         Answer("a", score=1.0))
    ]
    questions = [q for q, _ in data_set]

    resources = SharedResources(Vocab(), {"repr_dim": 100})
    example_reader = JTReader(resources,
                              SimpleMCInputModule(resources),
                              SimpleMCModelModule(resources),
                              SimpleMCOutputModule())

    # example_reader.setup_from_data(data_set)

    # todo: chose optimizer based on config
    example_reader.train(tf.train.AdamOptimizer(), data_set, max_epochs=10)

    answers = example_reader(questions)


def test_multi_support_fixed_class_inputs():
    import logging
    logging.basicConfig(level=logging.INFO)
    data_set = [
        (QASetting("Where is the cat?", ["the cat is on the mat."]), [Answer("mat")])
    ]
    shared_resources = SharedResources(Vocab(), {})
    input_module = MultiSupportFixedClassInputs()
    input_module.shared_resources = shared_resources
    input_module.setup_from_data(data_set)

    assert len(input_module.shared_resources.answer_vocab) == 1
    assert len(input_module.shared_resources.vocab) == 11

    tensor_data_set = list(input_module.batch_generator(data_set, False))

    expected_support = ["<SOS>", "the", "cat", "is", "on", "the", "mat", ".", "<EOS>"]
    expected_support_ids = [[[shared_resources.vocab.get_id(sym) for sym in expected_support]]]
    first_instance = tensor_data_set[0]
    actual_support_ids = first_instance[Ports.Input.multiple_support]
    assert np.array_equal(actual_support_ids, expected_support_ids)
    assert first_instance[Ports.Input.support_length][0] == len(expected_support)

    actual_answer_ids = first_instance[Ports.Target.target_index]
    expected_answer = [input_module.shared_resources.answer_vocab.get_id("mat")]
    assert np.array_equal(actual_answer_ids, expected_answer)

    actual_question_ids = first_instance[Ports.Input.question]
    expected_question = ["<SOS>", "where", "is", "the", "cat", "?", "<EOS>"]
    expected_question_ids = [[shared_resources.vocab.get_id(sym) for sym in expected_question]]
    assert np.array_equal(actual_question_ids, expected_question_ids)
    assert first_instance[Ports.Input.question_length][0] == len(expected_question)



    # print(tensor_data_set)


test_multi_support_fixed_class_inputs()
