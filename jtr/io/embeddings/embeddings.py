# -*- coding: utf-8 -*-
import logging
import pickle

import sys

from jtr.io.embeddings.word_to_vec import load_word2vec
from jtr.io.embeddings.glove import load_glove
from jtr.io.embeddings.fasttext import load_fasttext
import zipfile
import numpy as np


class Embeddings:
    """Wraps Vocabulary and embedding matrix to do lookups"""

    def __init__(self, vocabulary, lookup, filename: str = None, emb_format: str = None):
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
        return load_memory_map(file)


def load_memory_map(file_prefix: str) -> Embeddings:
    """
    Loads embeddings from a memory map file to allow lazy loading (and reduce the memory usage).
    Args:
        file_prefix: a file prefix. This function stores several files, and they will all start with this prefix.

    Returns:
        Embeddings object with a lookup matrix that is backed by a memory map.

    """
    meta_file = file_prefix + "_meta.pkl"
    mem_map_file = file_prefix + "_memmap"
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)
    shape = meta['shape']
    mem_map = np.memmap(mem_map_file, dtype='float32', mode='r+', shape=shape)
    result = Embeddings(meta['vocab'], mem_map, filename=file_prefix, emb_format="mem_map")
    return result


def save_as_memory_map(file_prefix: str, emb: Embeddings):
    meta_file = file_prefix + "_meta.pkl"
    mem_map_file = file_prefix + "_memmap"
    with open(meta_file, "wb") as f:
        pickle.dump({
            "vocab": emb.vocabulary,
            "shape": emb.shape
        }, f)
    mem_map = np.memmap(mem_map_file, dtype='float32', mode='w+', shape=emb.shape)
    mem_map[:] = emb.lookup[:]
    mem_map.flush()
    del mem_map


if __name__ == "__main__":
    input_name = sys.argv[1]
    output_prefix = sys.argv[2]
    embeddings = load_embeddings(input_name)
    logging.info("Loaded embeddings from {}".format(input_name))
    save_as_memory_map(output_prefix, embeddings)
    logging.info("Stored embeddings to {}".format(output_prefix))

