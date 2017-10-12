# -*- coding: utf-8 -*-

import numpy as np

from jack.util import map
from jack.util import preprocessing

text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et ' \
       'dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ' \
       'ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat ' \
       'nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit ' \
       'anim id est laborum.'

tokenized_text = ['Lorem', 'ipsum', 'dolor', 'sit', 'amet', ',', 'consectetur', 'adipiscing', 'elit', ',', 'sed',
                  'do', 'eiusmod', 'tempor', 'incididunt', 'ut', 'labore', 'et', 'dolore', 'magna', 'aliqua', '.',
                  'Ut', 'enim', 'ad', 'minim', 'veniam', ',', 'quis', 'nostrud', 'exercitation', 'ullamco',
                  'laboris', 'nisi', 'ut', 'aliquip', 'ex', 'ea', 'commodo', 'consequat', '.', 'Duis', 'aute',
                  'irure', 'dolor', 'in', 'reprehenderit', 'in', 'voluptate', 'velit', 'esse', 'cillum', 'dolore',
                  'eu', 'fugiat', 'nulla', 'pariatur', '.', 'Excepteur', 'sint', 'occaecat', 'cupidatat', 'non',
                  'proident', ',', 'sunt', 'in', 'culpa', 'qui', 'officia', 'deserunt', 'mollit', 'anim', 'id',
                  'est', 'laborum', '.']


def test_tokenize():
    assert preprocessing.tokenize(text) == tokenized_text
    question_text = "where is the cat?"
    desired_tokenised_question = ["where","is","the","cat","?"]
    assert preprocessing.tokenize(question_text) == desired_tokenised_question


def test_get_list_shape():
    data = [[1, 2, 3], [4, 5]]
    assert map.get_list_shape(data) == [2, 3]

    data = [[[1, 2, 3]], [[4, 5], [6, 7]]]
    assert map.get_list_shape(data) == [2, 2, 3]


def test_numpify():
    def _fillna(xs):
        data = np.array(xs)
        lens = np.array([len(i) for i in data])
        mask = np.arange(lens.max()) < lens[:, None]
        out = np.zeros(mask.shape, dtype=data.dtype)
        out[mask] = np.concatenate(data)
        return out

    data = [[1, 2, 3], [4, 5], [6, 7, 8]]
    data_np = map.numpify(data)

    for a, b in zip([np.array(x) for x in data], data_np):
        assert (a == b).all()

    data = {0: [[1, 2, 3]], 1: [[4, 5], [6, 7, 8]], 2: [[6, 7, 8]]}
    data_np = map.numpify(data)

    for ak, bk in zip(data.keys(), data_np.keys()):
        a, b = data[ak], data_np[bk]
        assert (_fillna(a) == b).all()
