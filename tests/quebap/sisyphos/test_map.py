# -*- coding: utf-8 -*-

from quebap.sisyphos import map

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
    assert map.tokenize(text) == tokenized_text


def test_lower():
    lower_text = [word.lower() for word in tokenized_text]
    assert map.lower(tokenized_text) == lower_text


def test_deep_map():
    a_lst = [[1, 2, 3], [4, 5, 6]]
    a_lst_map = map.deep_map(a_lst, lambda x: '_{}'.format(x))
    assert a_lst_map == [['_1', '_2', '_3'], ['_4', '_5', '_6']]

    a_dict = [{1: 0, 2: 1, 3: 0}, {4: 1, 5: 0, 6: 1}]
    a_dict_map = map.deep_map(a_dict, lambda x: '_{}'.format(x))
    assert a_dict_map == [{1: '_0', 2: '_1', 3: '_0'}, {4: '_1', 5: '_0', 6: '_1'}]


def test_dynamic_subsample():
    data = {
        'answers': [[1, 2], [3, 4]],
        'candidates': [range(0, 100), range(0, 100)]
    }
    data_ss = map.dynamic_subsample(xs=data, candidate_key='candidates', answer_key='answers', how_many=2, seed=0)
    assert data_ss['answers'] == [[1, 2], [3, 4]]
    assert list(data_ss['candidates'][0]) == [[1, 2], 49, 97]
    assert list(data_ss['candidates'][1]) == [[3, 4], 53, 5]


def test_get_list_shape():
    data = [[1, 2, 3], [4, 5]]
    assert map.get_list_shape(data) == [2, 3]

    data = [[[1, 2, 3]], [[4, 5], [6, 7]]]
    assert map.get_list_shape(data) == [2, 2, 3]


def test_get_seq_depth():
    data = [[1, 2, 3], [4, 5]]
    assert map.get_seq_depth(data) == [n - 1 for n in [2, 3]]

    data = [[[1, 2, 3]], [[4, 5], [6, 7]]]
    assert map.get_seq_depth(data) == [n - 1 for n in [2, 2, 3]]


def test_get_entry_dims():
    data = [[1, 2, 3], [4, 5], [6, 7, 8]]
    assert map.get_entry_dims(data) == [1, 1, 1]

    data = {2: 0, 3: [1, 2], 4: 2}
    assert map.get_entry_dims(data) == {2: 0, 3: 1, 4: 0}


def test_numpiify():
    pass
