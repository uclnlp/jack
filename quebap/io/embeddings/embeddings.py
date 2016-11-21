class Embeddings:
    def __init__(self, vocabulary, lookup):
        self.vocabulary = vocabulary
        self.lookup = lookup

    def get(self, word):
        id = self.vocabulary.get_idx_by_word(word)
        #out-of-vocab word
        if id is None:
            return None #lookup[None] would return entire lookup table
        #known embedding
        else:
            return self.lookup[id]

    @property
    def shape(self):
        return self.lookup.shape
