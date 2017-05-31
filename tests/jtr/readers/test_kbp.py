# -*- coding: utf-8 -*-

from jtr.data_structures import load_labelled_data


def test_kbp():
    data = load_labelled_data('tests/test_data/WN18/wn18-snippet.jtr.json')
    