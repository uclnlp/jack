import itertools

class Vocabulary:
    """Manages word2idx and idx2word functionality; manages of word stats."""
    def __init__(self, vocab=None):
        self.__word2idx = {}
        self.__freqs = {}
        if vocab is not None and isinstance(vocab, dict):
            self.__word2idx = {key: value for key, value in vocab.items()}
            self.__freqs = {key: None for key, _ in vocab.items()}
        self.__idx2word = None

    def __str__(self):
        return 'Vocabulary size: ' + str(self.get_size()) + \
               ' First 5 elements: ' + str(list(itertools.islice(self.__word2idx.keys(), 0, 5)))

    def add_iterable(self, itr):
        last = -1
        for item in itr:
            last = self.add_word(item)
        return last

    def add_word(self, word):
        if word not in self.__word2idx:
            self.__word2idx[word] = len(self.__word2idx)
            self.__freqs[word] = 1
            self.__idx2word = None
            return len(self.__word2idx) - 1
        else:
            self.__freqs[word] += 1
            return self.__word2idx[word]

    def get_idx_by_word(self, word):
        return self.__word2idx.get(word, None)

    def get_word_count(self, word):
        return self.__freqs.get(word, 0)

    def get_size(self):
        return len(self.__word2idx)

    def __len__(self):
        return len(self.__word2idx)

    def get_all_words(self):
        return self.__word2idx.keys()

    @property
    def word2idx(self):
        return self.__word2idx

    @property
    def idx2word(self):
        if self.__idx2word is None:
            self.__idx2word = [None] * len(self.__word2idx)
            for word, idx in self.__word2idx.items():
                self.__idx2word[idx] = word
        return self.__idx2word

    def get_word_by_idx(self, idx):
        if idx < len(self.idx2word):
            return self.idx2word[idx]
        else:
            return None

    def dump_all_tokens_to_file(self, filename):
        with open(filename, 'w') as f:
            for token in self.__word2idx.keys():
                f.write(token + '\n')

    def contains_word(self, word):
        return word in self.__word2idx

    def diff(self, other):
        if not isinstance(other, Vocabulary):
            raise TypeError()
        return set(self.__word2idx.keys()).difference(set(other.get_all_words()))

    def sorted(self):
        return sorted(self.__word2idx.items(), key=lambda x: -x[1][1])
