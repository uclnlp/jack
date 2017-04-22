# -*- coding: utf-8 -*-


def text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" "):
    if lower:
        text = text.lower()
    text = text.translate(str.maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [i for i in seq if i]


def one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
    seq = text_to_word_sequence(text, filters=filters,  lower=lower, split=split)
    return [(abs(hash(w)) % (n - 1) + 1) for w in seq]


class Tokenizer(object):
    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True, split=' ', char_level=False):
        self.word_counts = {}
        self.word_docs = {}
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = 0
        self.char_level = char_level
        self.word_index, self.index_docs = None, None

    def fit_on_texts(self, texts):
        self.document_count = 0
        for text in texts:
            self.document_count += 1
            seq = text if self.char_level else text_to_word_sequence(text, self.filters, self.lower, self.split)
            for w in seq:
                if w not in self.word_counts:
                    self.word_counts[w] = 0
                self.word_counts[w] += 1
            for w in set(seq):
                if w not in self.word_docs:
                    self.word_docs[w] = 0
                self.word_docs[w] += 1

        word_counts = list(self.word_counts.items())
        word_counts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in word_counts]
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        self.index_docs = {}
        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences(self, texts):
        return [seq for seq in self.texts_to_sequences_generator(texts)]

    def texts_to_sequences_generator(self, texts):
        num_words = self.num_words
        for text in texts:
            word_seq = text if self.char_level else text_to_word_sequence(text, self.filters, self.lower, self.split)
            idx_lst = []
            for w in word_seq:
                idx = self.word_index.get(w)
                if idx is not None and (not num_words or idx < num_words):
                    idx_lst += [idx]
            yield idx_lst
