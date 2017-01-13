# -*- coding: utf-8 -*-

import os
import tempfile
from quebap.load import read_quebap


def test_quebap_load():
    quebap_path = './quebap/data/SNLI/snippet_quebapformat_v1.json'
    if os.path.isfile(quebap_path):
        with open(quebap_path, 'r') as f:
            res = read_quebap.quebap_load(f, questions='single', supports='single', candidates='fixed', answers='single')

        print(res)
