# -*- coding: utf-8 -*-

import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_glove(stream, vocab=None):
    """Loads GloVe file and merges it if optional vocabulary
    Args:
        stream (iterable): An opened filestream to the GloVe file.
        vocab (dict=None): Word2idx dict of existing vocabulary.
    Returns:
        return_vocab (Vocabulary), lookup (matrix); Vocabulary contains the
                     word2idx and the matrix contains the embedded words.
    """
    logger.info('Loading GloVe vectors ..')

    word2idx = {}
    first_line = stream.readline()
    dim = len(first_line.split()) - 1
    lookup = np.empty([500000, dim], dtype=np.float)
    lookup[0] = np.fromstring(first_line.split(maxsplit=1)[1], sep=' ')
    word2idx[first_line.split(maxsplit=1)[0].decode('utf-8')] = 0
    n = 1
    for line in stream:
        word, vec = line.rstrip().split(maxsplit=1)
        if vocab is None or word in vocab and word not in word2idx:
            word = word.decode('utf-8')
            idx = len(word2idx)
            word2idx[word] = idx
            if idx > np.size(lookup, axis=0) - 1:
                lookup.resize([lookup.shape[0] + 500000, lookup.shape[1]])
            lookup[idx] = np.fromstring(vec, sep=' ')
        n += 1
    lookup.resize([len(word2idx), dim])
    logger.info('Loading GloVe vectors completed.')
    return word2idx, lookup
