from collections import defaultdict
import re
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
from sisyphos.vocab import Vocab


def tokenize(seq, pattern="([\s'\-\.\!])"):
    return [x for x in re.split(pattern, seq)
            if not re.match("\s", x) and x != ""]


def lower(seq):
    return [x.lower() for x in seq]


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
                if isinstance(xss, list) and all(
                        [not isinstance(xs, list) for xs in xss]):
                    xss_mapped.append(fun(xss))
                else:
                    xss_mapped.append(deep_seq_map(xs, fun))
            else:
                xss_mapped.append(xs)
        return xss_mapped


def get_list_shape(xs):
    shape = []
    if isinstance(xs, list):
        dim1 = len(xs)
        shape.append(dim1)
        if isinstance(xs[0], list):
            dims = [get_list_shape(x) for x in xs]
            for dim in max(dims):
                shape.append(dim)
    return shape


def map_to_numpy(xs, pad=0, indices=None):
    xs_np = []
    for i, x in enumerate(xs):
        if indices is None or i in indices:
            shape = get_list_shape(x)
            print(shape)


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
        ]
    ]

    vocab = Vocab()

    print(data)
    data_tokenized = deep_map(data, tokenize)
    data_lower = deep_seq_map(data_tokenized, lower)
    data_ids = deep_map(data_lower, vocab)
    data_ids_with_lengths = deep_seq_map(data_ids, lambda xs: len(xs), expand=True)
    print(data_tokenized)
    print(data_lower)
    print(data_ids)
    print(data_ids_with_lengths)
    print(vocab.get_id("afraid"))
    print(vocab.get_id("hal-9000"))  # <UNK>
    data_words = deep_map(data_ids_with_lengths, vocab.get_sym, indices=[0, 2])
    print(data_words)

    print()
    print(map_to_numpy(data_ids_with_lengths))

