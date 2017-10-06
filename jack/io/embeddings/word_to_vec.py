# -*- coding: utf-8 -*-

import gzip
import numpy as np

import logging

logger = logging.getLogger(__name__)


def load_word2vec(filename, vocab=None, normalise=True):
    """Loads a word2vec file and merges existing vocabulary.

    Args:
        filename (string): Path to the word2vec file.
        vocab (Vocabulary=None): Existing vocabulary to be merged.
        normalise (bool=True): If the word embeddings should be unit
                  normalized or not.
    Returns:
        return_vocab (dict), lookup (matrix): The dict is a word2idx dict and
        the lookup matrix is the matrix of embedding vectors.
    """
    logger.info("Loading word2vec vectors ..")
    with gzip.open(filename, 'rb') as f:
        vec_n, vec_size = map(int, f.readline().split())
        byte_size = vec_size * 4
        lookup = np.empty([vocab.get_size() if vocab is not None else vec_n, vec_size], dtype=np.float32)
        word2idx = {}
        idx = 0
        for n in range(vec_n):
            word = b''
            while True:
                c = f.read(1)
                if c == b' ':
                    break
                else:
                    word += c

            word = word.decode('utf-8')
            vector = np.fromstring(f.read(byte_size), dtype=np.float32)
            if vocab is None or vocab.contains_word(word):
                word2idx[word] = idx
                lookup[idx] = _normalise(vector) if normalise else vector
                idx += 1

    lookup.resize([idx, vec_size])
    logger.info('Loading word2vec vectors completed.')
    return word2idx, lookup


def _normalise(x):
    """Unit normalize x with L2 norm."""
    return (1.0 / np.linalg.norm(x, ord=2)) * x


def get_word2vec_vocabulary(fname):
    """Loads word2vec file and returns the vocabulary as dict word2idx."""
    voc, _ = load_word2vec(fname)
    return voc


if __name__ == "__main__":
    pickle_tokens = False
    vocab, _ = load_word2vec('../../data/word2vec/GoogleNews-vectors-negative300.bin.gz')

    # pickle token set
    if pickle_tokens:
        import pickle
        w2v_words = set(vocab.get_all_words())
        pickle.dump(w2v_words, open('./data/w2v_tokens.pickle', 'wb'))
