# -*- coding: utf-8 -*-

import os
from quebap.load import read_quebap


def test_quebap_load():
    quebap_path = './quebap/data/SNLI/snippet_quebapformat_v1.json'
    if os.path.isfile(quebap_path):
        with open(quebap_path, 'r') as f:
            res = read_quebap.quebap_load(f, questions='single', supports='single',
                                          candidates='fixed',answers='single')

        assert set(res['answers']) == {'neutral', 'contradiction', 'entailment', 'neutral', 'entailment',
                                       'contradiction', 'contradiction', 'entailment', 'neutral', 'neutral'}
        assert 'A person on a horse jumps over a broken down airplane.' in res['support']
        assert 'A person is training his horse for a competition.' in res['question']
