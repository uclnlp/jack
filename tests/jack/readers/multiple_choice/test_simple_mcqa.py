# -*- coding: utf-8 -*-

from jack.readers.classification.shared import *

from jack.util.vocab import Vocab


def test_single_support_fixed_class_inputs():
    import logging
    logging.basicConfig(level=logging.INFO)
    data_set = [
        (QASetting("Where is the cat?", ["the cat is on the mat."]), [Answer("mat")])
    ]
    shared_resources = SharedResources(Vocab(), {})
    input_module = ClassificationSingleSupportInputModule(shared_resources)
    input_module.setup_from_data(data_set)
    input_module.setup()

    assert len(input_module.shared_resources.answer_vocab) == 1
    assert len(input_module.shared_resources.vocab) == 9

    tensor_data_set = list(input_module.batch_generator(data_set, batch_size=3, is_eval=False))

    expected_support = ["the", "cat", "is", "on", "the", "mat", "."]
    expected_support_ids = [[shared_resources.vocab.get_id(sym) for sym in expected_support]]
    first_instance = tensor_data_set[0]
    actual_support_ids = first_instance[Ports.Input.support]
    assert np.array_equal(actual_support_ids, expected_support_ids)
    assert first_instance[Ports.Input.support_length][0] == len(expected_support)

    actual_answer_ids = first_instance[Ports.Target.target_index]
    expected_answer = [input_module.shared_resources.answer_vocab.get_id("mat")]
    assert np.array_equal(actual_answer_ids, expected_answer)

    actual_question_ids = first_instance[Ports.Input.question]
    expected_question = ["where", "is", "the", "cat", "?"]
    expected_question_ids = [[shared_resources.vocab.get_id(sym) for sym in expected_question]]
    assert np.array_equal(actual_question_ids, expected_question_ids)
    assert first_instance[Ports.Input.question_length][0] == len(expected_question)
