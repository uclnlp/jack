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
                 lower=True, split=' ', char_level=False, **kwargs):
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        self.word_counts = {}
        self.word_docs = {}
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = 0
        self.char_level = char_level

    def fit_on_texts(self, texts):
        self.document_count = 0
        for text in texts:
            self.document_count += 1
            seq = text if self.char_level else text_to_word_sequence(text, self.filters, self.lower, self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        self.index_docs = {}
        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences(self, texts):
        res = [vect for vect in self.texts_to_sequences_generator(texts)]
        return res

    def texts_to_sequences_generator(self, texts):
        num_words = self.num_words
        for text in texts:
            seq = text if self.char_level else text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        continue
                    else:
                        vect.append(i)
            yield vect
