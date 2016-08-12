from quebap.util.vocabulary import Vocabulary
import numpy as np


def load_glove(stream, vocab=None):
    print('[Loading GloVe]')
    word2idx = {}
    first_line = stream.readline()
    dim = len(first_line.split()) - 1
    lookup = np.empty([500000, dim], dtype=np.float)
    lookup[0] = np.fromstring(first_line.split(maxsplit=1)[1], sep=' ')
    word2idx[first_line.split(maxsplit=1)[0]] = 0
    n = 1
    idx = 1
    for line in stream:
        word, vec = line.rstrip().split(maxsplit=1)
        if vocab is None or word in vocab:
#            word = word.decode('utf-8')
            word2idx[word] = idx
            if idx > np.size(lookup, axis=0) - 1:
                lookup.resize([lookup.shape[0] + 500000, lookup.shape[1]])
            lookup[idx] = np.fromstring(vec, sep=' ')
            idx += 1
        n += 1
        if n % 100000 == 0:
            print('  ' + str(n // 1000) + 'k vectors processed...\r')
    lookup.resize([idx, dim])
    return_vocab = Vocabulary(word2idx)
    print('[Loading GloVe DONE]')
    return return_vocab, lookup

# The above method assumes an iterable of words, and returns the embeddings with
# a Vocabulary that is constructed in the loading.  Below is for when you already
# have such a Vocabulary object.
def load_wordreps_from_vocabulary(stream, vocab):
    print('[Loading Word Embeddings...]')
    first_line = stream.readline()
    dim = len(first_line.split()) - 1
    lookup = np.empty([500000, dim], dtype=np.float)
    lookup[0] = np.fromstring(first_line.split(maxsplit=1)[1], sep=' ')
    word2idx[first_line.split(maxsplit=1)[0]] = 0
    n = 1
    idx = 1
    for line in stream:
        word, vec = line.rstrip().split(maxsplit=1)
        if vocab is None or word in vocab:
            word = word.decode('utf-8')
            word2idx[word] = idx
            if idx > np.size(lookup, axis=0) - 1:
                lookup.resize([lookup.shape[0] + 500000, lookup.shape[1]])
            lookup[idx] = np.fromstring(vec, sep=' ')
            idx += 1
        n += 1
        if n % 100000 == 0:
            print('  ' + str(n // 1000) + 'k vectors processed...\r')
    lookup.resize([idx, dim])
    return_vocab = Vocabulary(word2idx)
    print('[Loading GloVe DONE]')
    return return_vocab, lookup


if __name__ == "__main__":
    pickle_tokens = False

    import zipfile

    with zipfile.ZipFile('../data/GloVe/glove.840B.300d.zip') as zf:
        with zf.open('glove.840B.300d.txt', 'r') as f:
            vocab, lookup = load_glove(f)

            # pickle token set
            if pickle_tokens:
                import pickle
                glove_words = set(vocab.get_all_words())
                pickle.dump(glove_words, open('./data/glove_tokens.pickle', 'wb'))
