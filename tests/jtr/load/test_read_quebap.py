# -*- coding: utf-8 -*-

import os
from jtr.load import read_jtr


def test_jtr_load():
    jtr_path = './jtr/data/SNLI/snippet_jtrformat_v1.json'
    if os.path.isfile(jtr_path):
        with open(jtr_path, 'r') as f:
            res = read_jtr.jtr_load(f, questions='single', supports='single',
                                          candidates='fixed',answers='single')

        assert set(res['answers']) == {'neutral', 'contradiction', 'entailment', 'neutral', 'entailment',
                                       'contradiction', 'contradiction', 'entailment', 'neutral', 'neutral'}
        assert 'A person on a horse jumps over a broken down airplane.' in res['support']
        assert 'A person is training his horse for a competition.' in res['question']
