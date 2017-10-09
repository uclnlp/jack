# -*- coding: utf-8 -*-

import tensorflow as tf

import jack.readers as readers
from jack.io.load import loaders


def test_kbp():
    data = loaders['jack']('tests/test_data/WN18/wn18-snippet.jack.json')
    questions = [question for question, _ in data]

    for model_name in ['transe', 'distmult', 'complex']:

        with tf.variable_scope(model_name):
            config = {
                'batch_size': 1,
                'repr_dim': 10
            }

            reader = readers.readers['{}_reader'.format(model_name)](config)
            reader.setup_from_data(data)

            answers = reader(questions)

            assert len(answers) == 5000

            assert answers, 'KBP reader should produce answers'
