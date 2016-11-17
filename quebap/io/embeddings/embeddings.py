class Embeddings:
    def __init__(self, vocabulary, lookup):
        self.vocabulary = vocabulary
        self.lookup = lookup

    def get(self, word):
        return self.lookup[self.vocabulary.get_idx_by_word(word)]

    @property
    def shape(self):
        return self.lookup.shape
