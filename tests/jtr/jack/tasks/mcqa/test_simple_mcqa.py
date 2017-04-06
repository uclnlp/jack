# -*- coding: utf-8 -*-

from jtr.jack.tasks.mcqa.simple_mcqa import *
from jtr.preprocess.vocab import Vocab

import tensorflow as tf

import pytest


@pytest.mark.skip("Not implemented yet")
def test_simple_mcqa():
    data_set = [
        (QASettingWithDefaults("which is it?", ["a is true", "b isn't"], atomic_candidates=["a", "b", "c"]),
         AnswerWithDefault("a", score=1.0))
    ]
    questions = [q for q, _ in data_set]

    resources = SharedVocabAndConfig(Vocab(), {"repr_dim": 100})
    example_reader = JTReader(resources,
                              SimpleMCInputModule(resources),
                              SimpleMCModelModule(resources),
                              SimpleMCOutputModule())

    # example_reader.setup_from_data(data_set)

    # todo: chose optimizer based on config
    example_reader.train(tf.train.AdamOptimizer(), data_set, max_epochs=10)

    answers = example_reader(questions)
