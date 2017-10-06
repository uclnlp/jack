# -*- coding: utf-8 -*-

import zipfile

from jack.io.embeddings.fasttext import load_fasttext
from jack.io.embeddings.glove import load_glove
from jack.io.embeddings.word_to_vec import load_word2vec


class Embeddings:
    """Wraps Vocabulary and embedding matrix to do lookups"""

    def __init__(self, vocabulary: dict, lookup, filename: str = None, emb_format: str = None):
        """
        Args:
            vocabulary:
            lookup:
            filename:
        """
        self.filename = filename
        self.vocabulary = vocabulary
        self.lookup = lookup
        self.emb_format = emb_format

    def get(self, word):
        _id = None
        if self.vocabulary is not None:
            _id = self.vocabulary.get(word, None)
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

    Args:
        file: string, path to a file like "GoogleNews-vectors-negative300.bin.gz" or "glove.42B.300d.zip"
        typ: string, either "word2vec", "glove", "fasttext" or "mem_map"
        options: dict, other options.
    Returns:
        Embeddings object, wrapper class around Vocabulary embedding matrix.
    """
    assert typ in {"word2vec", "glove", "fasttext", "mem_map"}, "so far only 'word2vec' and 'glove' foreseen"

    if typ.lower() == "word2vec":
        return Embeddings(*load_word2vec(file, **options))

    elif typ.lower() == "glove":
        if file.endswith('.txt'):
            with open(file, 'rb') as f:
                return Embeddings(*load_glove(f), filename=file, emb_format=typ)
        elif file.endswith('.zip'):
            with zipfile.ZipFile(file) as zf:
                txtfile = file.split('/')[-1][:-4] + '.txt'
                with zf.open(txtfile, 'r') as f:
                    return Embeddings(*load_glove(f), filename=file, emb_format=typ)
        else:
            raise NotImplementedError

    elif typ.lower() == "fasttext":
        with open(file, 'rb') as f:
            return Embeddings(*load_fasttext(f), filename=file, emb_format=typ)

    elif typ.lower() == "mem_map":
        from jack.io.embeddings.memory_map import load_memory_map
        return load_memory_map(file)
