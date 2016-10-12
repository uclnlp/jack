from collections import defaultdict
import re
import numpy as np
from pprint import pprint

# sym (e.g. token, token id or class label)
# seq (e.g. sequence of tokens)
# seqs (sequence of sequences)
# corpus (sequence of sequence of sequences)
#   e.g. hypotheses (sequence of sequences)
#        premises (sequence of sequences)
#        support (sequence of sequence of sequences)
#        labels (sequence of symbols)
# corpus = [hypotheses, premises, support, labels]
from sisyphos.vocab import Vocab, VocabEmb


def tokenize(xs, pattern="([\s'\-\.\!])"):
    return [x for x in re.split(pattern, xs)
            if not re.match("\s", x) and x != ""]


def lower(xs):
    return [x.lower() for x in xs]


def deep_map(xs, fun, indices=None, expand=False):
    """
    :param xs: a sequence of stuff
    :param fun: a function from x to something
    :return:
    """
    xs_mapped = []
    for i, x in enumerate(xs):
        if indices is None or i in indices:
            if expand:
                xs_mapped.append(x)
            if isinstance(x, list):
                x_mapped = deep_map(x, fun)
            else:
                x_mapped = fun(x)
            xs_mapped.append(x_mapped)
        else:
            xs_mapped.append(x)
    return xs_mapped


def deep_seq_map(xss, fun, indices=None, expand=False):
    """
    :param xss: a sequence of stuff
    :param fun: a function from xs to something
    :return:
    """
    if isinstance(xss, list) and all([not isinstance(xs, list) for xs in xss]):
        return fun(xss)
    else:
        xss_mapped = []
        for i, xs in enumerate(xss):
            if indices is None or i in indices:
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


def numpify(xs, pad=0, indices=None, dtypes=None):
    xs_np = []
    for i, x in enumerate(xs):
        if indices is None or i in indices:
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
                # todo: raise error
                pass
            xs_np.append(x_np)
        else:
            xs_np.append(x)
    return xs_np


if __name__ == '__main__':
    data = [
        [
            "All work and no play makes Jack a dull boy.",
            "All work and no play makes Jack a dull boy",
            "All work and no-play makes Jack a dull boy"
        ],
        [
            "I'm sorry Dave, I'm afraid I can't do that!",
            "I'm sorry Dave, I'm afraid I can't do that",
            "I'm sorry Dave, I'm afraid I can't do that"
        ],
        [  # support
            ["Play makes really dull", "really dull"],
            ["Dave is human"],
            ["All work", "all dull", "dull"]
        ]
    ]

    vocab = Vocab()

    print(data)
    data_tokenized = deep_map(data, tokenize)
    data_lower = deep_seq_map(data_tokenized, lower)
    data_ids = deep_map(data_lower, vocab)
    data_ids_with_lengths = deep_seq_map(data_ids, lambda xs: len(xs),
                                         indices=[0, 1, 2], expand=True)
    print(data_tokenized)
    print(data_lower)
    print(data_ids)
    print(data_ids_with_lengths)
    print(vocab.get_id("afraid"))
    print(vocab.get_id("hal-9000"))  # <UNK>
    data_words = deep_map(data_ids_with_lengths, vocab.get_sym, indices=[0, 2])
    print(data_words)

    print()
    print(numpify(data_ids_with_lengths))

