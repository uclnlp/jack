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
from quebap.sisyphos.vocab import Vocab, NeuralVocab



def tokenize(xs, pattern="([\s'\-\.\!])"):
    return [x for x in re.split(pattern, xs)
            if not re.match("\s", x) and x != ""]


def lower(xs):
    return [x.lower() for x in xs]


def deep_map(xs, fun, keys=None, fun_name=None, expand=False):
    """
    :param xs: a sequence or dict of stuff,
    :param fun: a function from x to something
    :param keys:  seq with keys if xs is dict; seq with integer indices if xs is seq
    :param fun_name: string with function tag (e.g. 'lengths'), used if expand==True and isinstance(xs,dict)
    :return:
    """

    if isinstance(xs,dict):
        xs_mapped = {}
        for k, x in xs.items():
            if keys is None or k in keys:
                if expand:
                    xs_mapped[k] = x
                    #if expand: create new key for transformed element, else use same key
                    k = '%s_%s'%(str(k),str(fun_name) if fun_name is not None else 'trf')
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
    """
    :param xss: a sequence or dict of stuff
    :param fun: a function from xs to something
    :param fun_name: string with function tag (e.g. 'lengths'), used if expand==True and isinstance(xs,dict)
    :return:
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
    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    dave = [
            "All work and no play makes Jack a dull boy",
            "All work and no play makes Jack a dull boy.",
            "All work and no play makes Jack a very dull boy!"]
    jack = [
            "I'm sorry Dave, I'm afraid I can't do that!",
            "I'm sorry Dave, I'm afraid I can't do that",
            "I'm sorry Dave, I'm afraid I cannot do that"]
    support = [
            ["Play makes really dull", "really dull"],
            ["Dave is human"],
            ["All work", "all dull", "dull"]]

    """test with seq"""
    data1 = [dave, jack, support]
    vocab1 = Vocab()
    data1_tokenized = deep_map(data1, tokenize)
    data1_lower = deep_seq_map(data1_tokenized, lower)
    data1_ids = deep_map(data1_lower, vocab1)
    data1_ids_with_lengths = deep_seq_map(data1_ids, lambda xs: len(xs),
                                         keys=[0, 1, 2], expand=True)
    print('\n(1) test encoded seq of seqs\n')
    pp.pprint(data1_ids_with_lengths)

    """test with dict"""
    print('\n(2) test encoded dict of seqs\n')
    data2 = {'dave': dave, 'jack': jack, 'support': support}
    vocab2 = Vocab()
    data2_tokenized = deep_map(data2, tokenize)
    data2_lower = deep_seq_map(data2_tokenized, lower)
    data2_ids = deep_map(data2_lower, vocab2)
    data2_ids_with_lengths = deep_seq_map(data2_ids, lambda xs: len(xs), keys=['dave','jack','support'],
                                          fun_name='lengths', expand=True)

    print('original dict:')
    pp.pprint(data2)
    print('lowercase tokenized encoded dict extended with lengths:')
    pp.pprint(data2_ids_with_lengths)

    print('test vocab:')
    print(vocab2.get_id("afraid"))
    print(vocab2.get_id("hal-9000"))  # <UNK>

    print('test words:')
    data2_words = deep_map(data2_ids_with_lengths, vocab2.get_sym, keys=['dave','jack'])
    print(data2_words)

    print('result from numpify:')
    print(numpify(data2_ids_with_lengths))

