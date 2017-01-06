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

