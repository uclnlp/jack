# -*- coding: utf-8 -*-

import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_fasttext(stream, vocab=None):
    """Loads fastText file and merges it if optional vocabulary
    Args:
        stream (iterable): An opened filestream to the fastText file.
        vocab (dict=None): Word2idx dict of existing vocabulary.
    Returns:
        return_vocab (Vocabulary), lookup (matrix); Vocabulary contains the
                     word2idx and the matrix contains the embedded words.
    """
    logger.info('Loading fastText vectors ..')

    word2idx = {}
    vec_n, vec_size = map(int, stream.readline().split())
    lookup = np.empty([vocab.get_size() if vocab is not None else vec_n, vec_size], dtype=np.float)
    n = 0
    for line in stream:
        word, vec = line.rstrip().split(maxsplit=1)
        if vocab is None or word in vocab and word not in word2idx:
            word = word.decode('utf-8')
            idx = len(word2idx)
            word2idx[word] = idx
            # if idx > np.size(lookup, axis=0) - 1:
            #    lookup.resize([lookup.shape[0] + 500000, lookup.shape[1]])
            lookup[idx] = np.fromstring(vec, sep=' ')
        n += 1
    # lookup.resize([len(word2idx), dim])
    logger.info('Loading fastText vectors completed.')
    return word2idx, lookup


if __name__ == "__main__":
    pickle_tokens = False

    import zipfile

    with zipfile.ZipFile('../data/GloVe/glove.840B.300d.zip') as zf:
        with zf.open('glove.840B.300d.txt', 'r') as f:
            from jack.io.embeddings import load_glove

            vocab, lookup = load_glove(f)

            # pickle token set
            if pickle_tokens:
                import pickle

                glove_words = set(vocab.get_all_words())
                pickle.dump(glove_words, open('./data/glove_tokens.pickle', 'wb'))
