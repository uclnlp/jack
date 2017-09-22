# -*- coding: utf-8 -*-

import tensorflow as tf

from jtr.core import SharedResources
import jtr.readers as readers
from jtr.data_structures import load_labelled_data


def test_kbp():
    data = load_labelled_data('tests/test_data/WN18/wn18-snippet.jtr.json')
    questions = [question for question, _ in data]

    for model_name in ['transe', 'distmult', 'complex']:
        with tf.variable_scope(model_name):
            config = {
                'batch_size': 1,
                'repr_dim': 10
            }

            shared_resources = SharedResources(None, config)
            reader = readers.get_reader_by_name('{}_reader'.format(model_name))
            reader.configure_with_shared_resources(shared_resources)
            reader.setup_from_data(data)

            answers = reader(questions)

            assert len(answers) == 5000

            assert answers, 'KBP reader should produce answers'
