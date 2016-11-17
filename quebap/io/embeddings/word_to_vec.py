import gzip
import numpy as np
from quebap.util.vocabulary import Vocabulary


def load_word2vec(filename, vocab=None, normalise=True):
    print("[Loading word2vec]")
    with gzip.open(filename, 'rb') as f:
        vec_n, vec_size = map(int, f.readline().split())
        byte_size = vec_size * 4
        if vocab is not None:
            lookup = np.empty([vocab.get_size(), vec_size], dtype=np.float32)
        else:
            lookup = np.empty([vec_n, vec_size], dtype=np.float32)
        word2idx = {}
        idx = 0
        for n in range(vec_n):
            if n % 100000 == 0 and n != 0:
                print('  ' + str(n // 1000) + 'k vectors processed...\r')
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
                if normalise:
                    lookup[idx] = normalize(vector)
                else:
                    lookup[idx] = vector
                idx += 1

    lookup.resize([idx, vec_size])
    return_vocab = Vocabulary(vocab=word2idx)
    return return_vocab, lookup


def normalize(x):
    return (1.0 / np.linalg.norm(x, ord=2)) * x


# def pickle_word2vec(vocab, lookup):
#     pickle.dump(vocab, open("word2vec_vocab.p", 'wb'))
#     pickle.dump(lookup, open("word2vec_lookup.p", 'wb'))
#
#
# def load_pickled_word2vec():
#     vocab = pickle.load(open('word2vec_vocab.p', 'rb'))
#     lookup = pickle.load(open('word2vec_lookup.p', 'rb'))
#     return vocab, lookup


def get_word2vec_vocabulary(fname):
    voc, _ = load_word2vec(fname)
    return voc


if __name__ == "__main__":
    pickle_tokens = False
    vocab, _ = load_word2vec('../data/word2vec/GoogleNews-vectors-negative300.bin.gz')

    # pickle token set
    if pickle_tokens:
        import pickle
        w2v_words = set(vocab.get_all_words())
        pickle.dump(w2v_words, open('./data/w2v_tokens.pickle', 'wb'))
