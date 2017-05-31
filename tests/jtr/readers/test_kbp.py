# -*- coding: utf-8 -*-

from jtr.core import SharedResources
import jtr.readers as readers
from jtr.data_structures import load_labelled_data


def test_distmult():
    data = load_labelled_data('tests/test_data/WN18/wn18-snippet.jtr.json')
    questions = [question for question, _ in data]

    config = {
        'batch_size': 1,
        'repr_dim': 10
    }

    shared_resources = SharedResources(None, config)
    distmult_reader = readers.readers['distmult_reader'](shared_resources)
    distmult_reader.setup_from_data(data)

    answers = distmult_reader(questions)

    assert len(answers) == 5000

    assert answers, 'KBP reader should produce answers'
