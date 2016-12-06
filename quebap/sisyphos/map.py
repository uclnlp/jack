from collections import defaultdict
import re
import numpy as np
import pprint
from quebap.sisyphos.vocab import Vocab


# sym (e.g. token, token id or class label)
# seq (e.g. sequence of tokens)
# seqs (sequence of sequences)
# corpus (sequence of sequence of sequences)
#   e.g. hypotheses (sequence of sequences)
#        premises (sequence of sequences)
#        support (sequence of sequence of sequences)
#        labels (sequence of symbols)
# corpus = [hypotheses, premises, support, labels]


def tokenize(xs, pattern="([\s'\-\.\,\!])"):
    return [x for x in re.split(pattern, xs)
            if not re.match("\s", x) and x != ""]

def lower(xs):
    """returns lowercase for sequence of strings"""
    #"""performs lowercasing on string or sequence of strings"""
    #if isinstance(xs, str):
    #    return xs.lower()
    return [x.lower() for x in xs]


def deep_map(xs, fun, keys=None, fun_name='trf', expand=False):
    """Performs deep mapping of the input `xs` using function `fun`.
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

    Returns:
      Transformed sequence or dictionary.

    Example:

    (1) Test with sequence of stuff
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
    >>> data1_ids_with_lengths = deep_seq_map(data1_ids, lambda xs: len(xs),
    ...                                       fun_name='lengths', expand=True)
    >>> pprint.pprint(data1_ids_with_lengths)
    [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
      [1, 2, 3, 4, 5, 6, 7, 8, 12, 9, 10, 13]],
     [10, 11, 12],
     [[14, 15, 16, 17, 18, 19, 14, 15, 16, 20, 14, 21, 15, 22, 23, 24, 13],
      [14, 15, 16, 17, 18, 19, 14, 15, 16, 20, 14, 21, 15, 22, 23, 24],
      [14, 15, 16, 17, 18, 19, 14, 15, 16, 20, 14, 25, 23, 24]],
     [17, 16, 14],
     [[[5, 6, 26, 9], [26, 9]], [[18, 27, 28]], [[1, 2], [1, 9], [9]]],
     [[4, 2], [3], [2, 2, 1]]]


    (2) Test with data dictionary
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

    if isinstance(xs, dict):
        xs_mapped = {}
        for k, x in sorted(xs.items(), key=lambda it:it[0]): #to make deterministic (e.g. for consistent symbol id's)
            if keys is None or k in keys:
                if expand:
                    xs_mapped[k] = x
                    #if expand: create new key for transformed element, else use same key
                    k = '%s_%s'%(str(k),str(fun_name))
                if isinstance(x, list) or isinstance(x, dict):
                    x_mapped = deep_map(x, fun)
                else:
                    x_mapped = fun(x)
                xs_mapped[k] = x_mapped
            else:
                xs_mapped[k] = x
    else:
        xs_mapped = []
        for k, x in enumerate(xs):
            if keys is None or k in keys:
                if expand:
                    xs_mapped.append(x)
                if isinstance(x, list) or isinstance(x, dict):
                    x_mapped = deep_map(x, fun, fun_name=fun_name)
                else:
                    x_mapped = fun(x)
                xs_mapped.append(x_mapped)
            else:
                xs_mapped.append(x)
    return xs_mapped




def deep_seq_map(xss, fun, keys=None, fun_name=None, expand=False):
    """Performs deep mapping of the input `xs` using function `fun`.
    In case `expand==False` each top-level entry of `xs` to be transformed
    replaces the original entry.
    `deep_map` supports `xs` to be a dictionary or a list/tuple:
      - In case `xs` is a dictionary, its transformed value is also a dictionary, and `keys` contains the keys of the
      values to be transformed.
      - In case `xs` is a list/tuple, `keys` contains the indices of the entries to be transformed
    The function `deep_map` is recursively applied to the values of `xs`;
    the function `fun` takes a sequence as input, and is applied at the one but deepest level,
    where the entries are sequences of objects (no longer sequences of sequences).
    This is the only difference with `deeo_map`

    Args:
      `xs`: a sequence (list/tuple) of objects or sequences of objects.
      `fun`: a function to transform sequences
      `keys`: seq with keys if `xs` is dict; seq with integer indices if `xs` is seq.
        For entries not in `keys`, the original `xs` value is retained.
      `fun_name`: default value 'trf'; string with function tag (e.g. 'lengths'),
        used if '''expand==True''' and '''isinstance(xs,dict)'''
        Say for example fun_name='count', and `keys` contains 'sentence', then the transformed dict would look like
        '''{'sentence':[sentences], 'sentence_lengths':[fun(sentences)] ...}'''

    Returns:
      Transformed sequence or dictionary.

    Example:
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
    >>> data2 = {'dave': dave, 'jack': jack, 'support': support}
    >>> vocab2 = Vocab()
    >>> data2_processed = deep_map(data2, lambda x: tokenize(x.lower()))
    >>> data2_ids = deep_map(data2_processed, vocab2)
    >>> data2_ids_with_lengths = deep_seq_map(data2_ids, lambda xs: len(xs), keys=['dave','jack','support'],
    ...                                       fun_name='lengths', expand=True)
    >>> pprint.pprint(data2_ids_with_lengths)
    {'dave': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
              [1, 2, 3, 4, 5, 6, 7, 8, 12, 9, 10, 13]],
     'dave_lengths': [10, 11, 12],
     'jack': [[14, 15, 16, 17, 18, 19, 14, 15, 16, 20, 14, 21, 15, 22, 23, 24, 13],
              [14, 15, 16, 17, 18, 19, 14, 15, 16, 20, 14, 21, 15, 22, 23, 24],
              [14, 15, 16, 17, 18, 19, 14, 15, 16, 20, 14, 25, 23, 24]],
     'jack_lengths': [17, 16, 14],
     'support': [[[5, 6, 26, 9], [26, 9]], [[18, 27, 28]], [[1, 2], [1, 9], [9]]],
     'support_lengths': [[4, 2], [3], [2, 2, 1]]}
    """

    if isinstance(xss, list) and all([not isinstance(xs, list) for xs in xss]):
        return fun(xss)
    else:
        if isinstance(xss, dict):
            xss_mapped = {}
            for k, xs in xss.items():
                if keys is None or k in keys:
                    if expand:
                        xss_mapped[k] = xs
                        k = '%s_%s'%(str(k), str(fun_name) if fun_name is not None else 'trf')
                    if isinstance(xs, list) and all([not isinstance(x, list) for x in xs]):
                        xss_mapped[k] = fun(xs)
                    else:
                        xss_mapped[k] = deep_seq_map(xs, fun) #fun_name not needed, because expand==False
                else:
                    xss_mapped[k] = xs
        else:
            xss_mapped = []
            for k, xs in enumerate(xss):
                if keys is None or k in keys:
                    if expand:
                        xss_mapped.append(xs)
                    if isinstance(xs, list) and all([not isinstance(x, list) for x in xs]):
                        xss_mapped.append(fun(xs))
                    else:
                        xss_mapped.append(deep_seq_map(xs, fun))
                else:
                    xss_mapped.append(xs)
        return xss_mapped




def get_list_shape(xs):
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
    return [n-1 for n in get_list_shape(xs)]


def numpify(xs, pad=0, keys=None, dtypes=None):
    is_dict = isinstance(xs, dict)
    xs_np = {} if is_dict else [0]*len(xs)
    xs_iter = xs.items() if is_dict else enumerate(xs)

    for key, x in xs_iter:
        if keys is None or key in keys:
            shape = get_list_shape(x)
            if dtypes is None:
                dtype = np.int64
            else:
                dtype = dtypes[i]
            x_np = np.full(shape, pad, dtype)
            dims = len(shape)
            if dims == 1:
                x_np[0:shape[0]] = x
            elif dims == 2:
                for j, y in enumerate(x):
                    x_np[j, 0:len(y)] = y
            elif dims == 3:
                for j, ys in enumerate(x):
                    for k, y in enumerate(ys):
                        x_np[j, k, 0:len(y)] = y
            else:
                raise(NotImplementedError)
                #todo: extend to general case
                pass
            xs_np[key] = x_np
        else:
            xs_np[key] = x
    return xs_np


if __name__ == '__main__':


    import doctest
    print(doctest.testmod())

