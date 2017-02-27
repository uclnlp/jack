from collections import defaultdict
import re
import numpy as np
import pprint
from jtr.preprocess.vocab import Vocab
from jtr.util.rs import DefaultRandomState

rs = DefaultRandomState(1337)#new seed ignored if set previously

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
    """Splits sentences into tokens by regex over punctuation: ( -.,!])["""
    return [x for x in re.split(pattern, xs)
            if not re.match("\s", x) and x != ""]

def notokenize(xs):
    """Embeds deepest itemns into a list"""
    return [xs]


def lower(xs):
    """returns lowercase for sequence of strings"""
    # """performs lowercasing on string or sequence of strings"""
    # if isinstance(xs, str):
    #    return xs.lower()
    return [x.lower() for x in xs]


def deep_map(xs, fun, keys=None, fun_name='trf', expand=False, cache_fun=False):
    """Applies fun to a dict or list; adds the results in-place.

    Usage: Transform a corpus iteratively by applying functions like
    `tokenize`, `lower`, or vocabulary functions (word -> embedding id) to it.
    ::
      from jtr.sisyphos.vocab import Vocab
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
                        x_mapped = deep_map_recursion(x) #deep_map(x, fun, fun_name=fun_name)
                    else:
                        x_mapped = fun(x)
                    xs_mapped.append(x_mapped)
                else:
                    xs_mapped.append(x)
        if cache_fun:
            cache[id(inner_xs)] = xs_mapped
        return xs_mapped

    return deep_map_recursion(xs,keys)


def deep_seq_map(xss, fun, keys=None, fun_name=None, expand=False):
    """Applies fun to list of or dict of lists; adds the results in-place.

    Usage: Transform a corpus iteratively by applying functions like
    `tokenize`, `lower`, or vocabulary functions (word -> embedding id) to it.

    from jtr.sisyphos.vocab import Vocab
    vocab = Vocab()
    keys = ['question', 'support']

    corpus = deep_map(corpus, lambda x: x.lower(), keys)
    corpus = deep_map(corpus, tokenize, keys)
    corpus = deep_map(corpus, vocab, keys)
    corpus = deep_map(corpus, vocab._normalize, keys=keys)
    -> through tokenize we go from a dict of sentences to
       a dict of words (list of lists), thus we now apply deep_seq_map for
       processing to add start of and end of sentence tags:
    corpus = deep_seq_map(corpus, lambda xs: ["<SOS>"] + xs +
                                             ["<EOS>"],
                                             ['question', 'support'])

    -> From here we can create batches from the corpus and feed it into a model.

    In case `expand==False` each top-level entry of `xs` to be transformed
    replaces the original entry.
    `deep_map` supports `xs` to be a dictionary or a list/tuple:
      - In case `xs` is a dictionary, its transformed value is also a dictionary, and `keys` contains the keys of the
      values to be transformed.
      - In case `xs` is a list/tuple, `keys` contains the indices of the entries to be transformed
    The function `deep_map` is recursively applied to the values of `xs`;
    the function `fun` takes a sequence as input, and is applied at the one but deepest level,
    where the entries are sequences of objects (no longer sequences of sequences).
    This is the only difference with `deep_map`

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
                        k = '%s_%s' % (str(k), str(fun_name) if fun_name is not None else 'trf')
                    if isinstance(xs, list) and all([not isinstance(x, list) for x in xs]):
                        xss_mapped[k] = fun(xs)
                    else:
                        xss_mapped[k] = deep_seq_map(xs, fun)  # fun_name not needed, because expand==False
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


def dynamic_subsample(xs, candidate_key, answer_key, how_many=1, avoid=[]):
    """Replaces candidates by a mix of answers and random candidates.

    Creates negative samples by combining the true answers and some random
    deletion of entries in the candidates. Then replaces the candidates
    dictionary and returns it.

    Replace a list of lists with a list of dynamically subsampled lists. The dynamic list will
    always contain the elements from the `answer_key` list, and a subsample of size `how_many` from
    the corresponding `candidate_key` list
    Args:
        xs: a dictionary of keys to lists
        candidate_key: the key of the candidate list
        answer_key: the key of the answer list
        how_many: how many samples from the candidate list should we take
        avoid: list of candidates to be avoided
        (note: only those are avoided, any instances according to `answer_key` which are not
        in `avoid`, may still be sampled!)

    Returns:
        a new dictionary identical to `xs` for all but the `candidate_key`. For that key the value
        is a list of `DynamicSubsampledList` objects.

    Example:
        >>> data = {'answers':[[1,2],[3,4]], 'candidates': [range(0,100), range(0,100)]}
        >>> processed = dynamic_subsample(data, 'candidates', 'answers', 2)
        >>> " | ".join([" ".join([str(elem) for elem in elems]) for elems in processed['candidates']])
        '1 2 89 39 | 3 4 90 82'
        >>> " | ".join([" ".join([str(elem) for elem in elems]) for elems in processed['candidates']])
        '1 2 84 72 | 3 4 9 6'
        >>> " | ".join([" ".join([str(elem) for elem in elems]) for elems in processed['answers']])
        '1 2 | 3 4'
        >>> processed = dynamic_subsample(data, 'candidates', 'answers', 5, avoid=range(91))
        >>> " | ".join([" ".join([str(elem) for elem in elems]) for elems in processed['candidates']])
        '1 2 93 91 91 95 97 | 3 4 93 99 92 98 93'
    """
    candidate_dataset = xs[candidate_key]
    answer_dataset = xs[answer_key]
    new_candidates = []
    assert (len(candidate_dataset) == len(answer_dataset))
    for i in range(0, len(candidate_dataset)):
        candidates = candidate_dataset[i]
        answers = [answer_dataset[i]] if not hasattr(answer_dataset[i],'__len__') else answer_dataset[i]
        new_candidates.append(DynamicSubsampledList(answers, candidates, how_many, avoid=[], rand=rs))
    result = {}
    result.update(xs)
    result[candidate_key] = new_candidates
    return result




class DynamicSubsampledList:
    """
    A container that produces different list subsamples on every call to `__iter__`.

    >>> dlist = DynamicSubsampledList([1,2], range(0,100),2, rand=rs)
    >>> print(" ".join([str(e) for e in dlist]))
    1 2 23 61
    >>> print(" ".join([str(e) for e in dlist]))
    1 2 92 39
    """

    def __init__(self, always_in, to_sample_from, how_many, avoid=[], rand=rs):
        self.always_in = always_in
        self.to_sample_from = to_sample_from
        self.how_many = how_many
        self.avoid = set(avoid)
        self.random = rand

    def __iter__(self):
        result = []
        result += self.always_in
        if len(self.avoid) == 0:
            result.extend(list(self.random.choice(self.to_sample_from, size=self.how_many, replace=True)))
        else:
            for _ in range(self.how_many):
                avoided = False
                trial, max_trial = 0, 50
                while (not avoided and trial < max_trial):
                    samp = self.random.choice(self.to_sample_from)
                    trial += 1
                    avoided = False if samp in self.avoid else True
                result.append(samp)
        return result.__iter__()

    def __len__(self):
        return len(self.always_in)+self.how_many#number of items is the number of answers plus number of negative samples
    
    def __getitem__(self, key):
        #todo: verify
        return self.always_in[0]


def get_list_shape(xs):
    if isinstance(xs,int):
        shape=[]
    else:
        shape = [len(xs)]
        for i, x in enumerate(xs):
            if isinstance(x, list) or isinstance(x, DynamicSubsampledList):
                if len(shape) == 1:
                    shape.append(0)
                shape[1] = max(len(x), shape[1])
                for j, y in enumerate(x):
                    if isinstance(y, list) or isinstance(y, DynamicSubsampledList):
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
    #todo: implement recursive form; now only OK for 'regular' (=most common type of) data structures
    if isinstance(corpus, dict):
        keys = list(corpus.keys())
        dims = {key: 0 for key in keys}
    else:
        keys = range(len(corpus))
        dims = [0 for i in range(len(corpus))]  #scalars have dim 0 (but tensor version will have shape length 1)
    for key in keys:
        entry = corpus[key]
        try:
            while hasattr(entry, '__len__'):
                dims[key] += 1
                entry = entry[0]  #will fail if entry is dict
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
            if dtypes is None:
                dtype = np.int64
            else:
                dtype = dtypes[i]
            x_np = np.full(shape, pad, dtype)
            dims = len(shape)
            if dims == 0:
                x_np=x
            elif dims == 1:
                x_np[0:shape[0]] = x
            elif dims == 2:
                for j, y in enumerate(x):
                    x_np[j, 0:len(y)] = [ys for ys in y]#this comprehension turns DynamicSubsampledList into a list
            elif dims == 3:
                for j, ys in enumerate(x):
                    for k, y in enumerate(ys):
                        x_np[j, k, 0:len(y)] = y
            else:
                raise (NotImplementedError)
                # todo: extend to general case
                pass
            xs_np[key] = x_np
        else:
            xs_np[key] = x
    return xs_np


if __name__ == '__main__':
    import doctest

    print(doctest.testmod())
