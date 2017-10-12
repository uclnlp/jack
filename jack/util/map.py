# -*- coding: utf-8 -*-

import pprint

import numpy as np

from jack.util.random import DefaultRandomState
from jack.util.vocab import Vocab

rs = DefaultRandomState(1337)


# sym (e.g. token, token id or class label)
# seq (e.g. sequence of tokens)
# seqs (sequence of sequences)
# corpus (sequence of sequence of sequences)
#   e.g. hypotheses (sequence of sequences)
#        premises (sequence of sequences)
#        support (sequence of sequence of sequences)
#        labels (sequence of symbols)
# corpus = [hypotheses, premises, support, labels]


def lower(xs):
    """returns lowercase for sequence of strings"""
    return [x.lower() for x in xs]


def deep_map(xs, fun, keys=None, fun_name='trf', expand=False, cache_fun=False):
    """Applies fun to a dict or list; adds the results in-place.

    Usage: Transform a corpus iteratively by applying functions like
    `tokenize`, `lower`, or vocabulary functions (word -> embedding id) to it.
    ::
      from jack.sisyphos.vocab import Vocab
      vocab = Vocab()
      keys = ['question', 'support']
      corpus = deep_map(corpus, lambda x: x.lower(), keys)
      corpus = deep_map(corpus, tokenize, keys)
      corpus = deep_map(corpus, vocab, keys)
      corpus = deep_map(corpus, vocab._normalize, keys=keys)

    From here we can create batches from the corpus and feed it into a model.

    In case `expand==False` each top-level entry of `xs` to be transformed
    replaces the original entry.
    `deep_map` supports `xs` to be a dictionary or a list/tuple:
      - In case `xs` is a dictionary, its transformed value is also a dictionary, and `keys` contains the keys of the
      values to be transformed.
      - In case `xs` is a list/tuple, `keys` contains the indices of the entries to be transformed
    The function `deep_map` is recursively applied to the values of `xs`,
    only at the deepest level, where the entries are no longer sequences/dicts, after which `fun` is applied.

    Args:
      `xs`: a sequence (list/tuple) of objects or sequences of objects.
      `fun`: a function to transform objects
      `keys`: seq with keys if `xs` is dict; seq with integer indices if `xs` is seq.
        For entries not in `keys`, the original `xs` value is retained.
      `fun_name`: default value 'trf'; string with function tag (e.g. 'lengths'),
        used if '''expand==True''' and '''isinstance(xs,dict)'''
        Say for example fun_name='lengths', and `keys` contains 'sentence', then the transformed dict would look like
        '''{'sentence':[sentences], 'sentence_lengths':[fun(sentences)] ...}'''
      `cache_fun`: should the function values for seen inputs be cached. Use with care, as it will affect functions with side effects.

    Returns:
      Transformed sequence or dictionary.

    Example:

    >>> #(1) Test with sequence of stuff
    >>> dave = [
    ...         "All work and no play makes Jack a dull boy",
    ...         "All work and no play makes Jack a dull boy.",
    ...         "All work and no play makes Jack a very dull boy!"]
    >>> jack = [
    ...         "I'm sorry Dave, I'm afraid I can't do that!",
    ...         "I'm sorry Dave, I'm afraid I can't do that",
    ...         "I'm sorry Dave, I'm afraid I cannot do that"]
    >>> support = [
    ...         ["Play makes really dull", "really dull"],
    ...         ["Dave is human"],
    ...         ["All work", "all dull", "dull"]]
    >>> data1 = [dave, jack, support]
    >>> vocab1 = Vocab()
    >>> data1_lower = deep_map(data1, lambda s:s.lower())
    >>> data1_tokenized = deep_map(data1_lower, tokenize)
    >>> data1_ids = deep_map(data1_tokenized, vocab1)
    >>> pprint.pprint(data1_ids)
    [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
      [1, 2, 3, 4, 5, 6, 7, 8, 12, 9, 10, 13]],
     [[14, 15, 16, 17, 18, 19, 14, 15, 16, 20, 14, 21, 15, 22, 23, 24, 13],
      [14, 15, 16, 17, 18, 19, 14, 15, 16, 20, 14, 21, 15, 22, 23, 24],
      [14, 15, 16, 17, 18, 19, 14, 15, 16, 20, 14, 25, 23, 24]],
     [[[5, 6, 26, 9], [26, 9]], [[18, 27, 28]], [[1, 2], [1, 9], [9]]]]

    >>> #(2) Test with data dictionary
    >>> data2 = {'dave': dave, 'jack': jack, 'support': support}
    >>> pprint.pprint(data2)
    {'dave': ['All work and no play makes Jack a dull boy',
              'All work and no play makes Jack a dull boy.',
              'All work and no play makes Jack a very dull boy!'],
     'jack': ["I'm sorry Dave, I'm afraid I can't do that!",
              "I'm sorry Dave, I'm afraid I can't do that",
              "I'm sorry Dave, I'm afraid I cannot do that"],
     'support': [['Play makes really dull', 'really dull'],
                 ['Dave is human'],
                 ['All work', 'all dull', 'dull']]}
    >>> data2_tokenized = deep_map(data2, tokenize)
    >>> pprint.pprint(data2_tokenized['support'])
    [[['Play', 'makes', 'really', 'dull'], ['really', 'dull']],
     [['Dave', 'is', 'human']],
     [['All', 'work'], ['all', 'dull'], ['dull']]]
    """

    cache = {}

    def deep_map_recursion(inner_xs, keys=None):
        if cache_fun and id(inner_xs) in cache:
            return cache[id(inner_xs)]
        if isinstance(inner_xs, dict):
            xs_mapped = {}
            for k, x in sorted(inner_xs.items(),
                               key=lambda it: it[0]):  # to make deterministic (e.g. for consistent symbol id's)
                if keys is None or k in keys:
                    if expand:
                        xs_mapped[k] = x
                        # if expand: create new key for transformed element, else use same key
                        k = '%s_%s' % (str(k), str(fun_name))
                    if isinstance(x, list) or isinstance(x, dict):
                        x_mapped = deep_map_recursion(x)
                    else:
                        x_mapped = fun(x)
                    xs_mapped[k] = x_mapped
                else:
                    xs_mapped[k] = x
        else:
            xs_mapped = []
            for k, x in enumerate(inner_xs):
                if keys is None or k in keys:
                    if expand:
                        xs_mapped.append(x)
                    if isinstance(x, list) or isinstance(x, dict):
                        x_mapped = deep_map_recursion(x)
                    else:
                        x_mapped = fun(x)
                    xs_mapped.append(x_mapped)
                else:
                    xs_mapped.append(x)
        if cache_fun:
            cache[id(inner_xs)] = xs_mapped
        return xs_mapped

    return deep_map_recursion(xs, keys)


def get_list_shape(xs):
    if isinstance(xs, int):
        shape = []
    else:
        shape = [len(xs)]
        for i, x in enumerate(xs):
            if isinstance(x, list):
                if len(shape) == 1:
                    shape.append(0)
                shape[1] = max(len(x), shape[1])
                for j, y in enumerate(x):
                    if isinstance(y, list):
                        if len(shape) == 2:
                            shape.append(0)
                        shape[2] = max(len(y), shape[2])
    return shape


def get_seq_depth(xs):
    return [n - 1 for n in get_list_shape(xs)]


def get_entry_dims(corpus):
    """
    get number of dimensions for each entry; needed for placeholder generation
    """
    # todo: implement recursive form; now only OK for 'regular' (=most common type of) data structures
    if isinstance(corpus, dict):
        keys = list(corpus.keys())
        dims = {key: 0 for key in keys}
    else:
        keys = range(len(corpus))
        dims = [0 for i in range(len(corpus))]  # scalars have dim 0 (but tensor version will have shape length 1)
    for key in keys:
        entry = corpus[key]
        try:
            while hasattr(entry, '__len__'):
                dims[key] += 1
                entry = entry[0]  # will fail if entry is dict
        except:
            dims[key] = None
    return dims


def numpify(xs, pad=0, keys=None, dtypes=None):
    """Converts a dict or list of Python data into a dict of numpy arrays."""
    is_dict = isinstance(xs, dict)
    xs_np = {} if is_dict else [0] * len(xs)
    xs_iter = xs.items() if is_dict else enumerate(xs)

    for i, (key, x) in enumerate(xs_iter):
        if keys is None or key in keys:
            shape = get_list_shape(x)
            dtype = dtypes[i] if dtypes is not None else np.int64
            x_np = np.full(shape, pad, dtype)

            nb_dims = len(shape)

            if nb_dims == 0:
                x_np = x

            else:
                def f(tensor, values):
                    t_shp = tensor.shape
                    if len(t_shp) > 1:
                        for _i, _values in enumerate(values):
                            f(tensor[_i], _values)
                    else:
                        tensor[0:len(values)] = [v for v in values]

                f(x_np, x)

            xs_np[key] = x_np
        else:
            xs_np[key] = x
    return xs_np


if __name__ == '__main__':
    import doctest

    print(doctest.testmod())
