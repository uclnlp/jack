# -*- coding: utf-8 -*-

from jtr.load.embeddings.word_to_vec import load_word2vec
from jtr.load.embeddings.glove import load_glove
import zipfile


class Embeddings:
    """Wraps Vocabulary and embedding matrix to do lookups"""
    def __init__(self, vocabulary, lookup):
        """

        :param vocabulary:
        :param lookup:
        """
        self.vocabulary = vocabulary
        self.lookup = lookup

    def get(self, word):
        _id = None
        if self.vocabulary is not None:
            _id = self.vocabulary.get_idx_by_word(word)
        # Handling OOV words - Note: lookup[None] would return entire lookup table
        return self.lookup[_id] if _id is not None else None

    def __call__(self, word):
        return self.get(word)

    @property
    def shape(self):
        return self.lookup.shape


def load_embeddings(file, typ='glove', **options):
    """
    Loads either GloVe or word2vec embeddings and wraps it into Embeddings

    :param file: string, path to a file like "GoogleNews-vectors-negative300.bin.gz" or "glove.42B.300d.zip"
    :param typ: string, either "word2vec" or "glove"
    :param options: dict, other options.
    :return: Embeddings object, wrapper class around Vocabulary embedding matrix.
    """
    assert typ in {"word2vec", "glove"}, "so far only 'word2vec' and 'glove' foreseen"

    if typ.lower() == "word2vec":
        return Embeddings(*load_word2vec(file, **options))

    elif typ.lower() == "glove":
        if file.endswith('.txt'):
            with open(file, 'rb') as f:
                return Embeddings(*load_glove(f))
        elif file.endswith('.zip'):
            with zipfile.ZipFile(file) as zf:
                txtfile = file.split('/')[-1][:-4]+'.txt'
                with zf.open(txtfile, 'r') as f:
                    return Embeddings(*load_glove(f))
        else:
            raise NotImplementedError
