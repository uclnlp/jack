# -*- coding: utf-8 -*-

import tensorflow as tf

from jack.core import SharedResources
import jack.readers as readers
from jack.data_structures import load_labelled_data


def test_kbp():
    data = load_labelled_data('tests/test_data/WN18/wn18-snippet.jack.json')
    questions = [question for question, _ in data]

    for model_name in ['transe', 'distmult', 'complex']:

        with tf.variable_scope(model_name):
            config = {
                'batch_size': 1,
                'repr_dim': 10
            }

            shared_resources = SharedResources(None, config)
            reader = readers.readers['{}_reader'.format(model_name)](shared_resources)
            reader.setup_from_data(data)

            answers = reader(questions)

            assert len(answers) == 5000

            assert answers, 'KBP reader should produce answers'
