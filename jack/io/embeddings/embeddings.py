# -*- coding: utf-8 -*-
import os
import pickle
import zipfile

import yaml

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

    def get(self, word, default=None):
        _id = None
        if self.vocabulary is not None:
            _id = self.vocabulary.get(word, None)
        # Handling OOV words - Note: lookup[None] would return entire lookup table
        return self.lookup[_id] if _id is not None else default

    def __call__(self, word):
        return self.get(word)

    @property
    def shape(self):
        return self.lookup.shape

    def store(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        if self.filename is None:
            self.filename = os.path.join(path, "emb.pkl")
            self.emb_format = 'pkl'
            with open(self.filename, "wb") as f:
                pickle.dump(self, f)
        conf_file = os.path.join(path, "config.yaml")
        with open(conf_file, "w") as f:
            yaml.dump({"embedding_file": self.filename, "emb_format": self.emb_format}, f)

    @staticmethod
    def from_config(conf_file):
        with open(conf_file, "r") as f:
            config = yaml.load(f)
        if config["embedding_file"] is not None:
            return load_embeddings(config["embedding_file"], typ=config.get("emb_format", None))

    @staticmethod
    def from_dir(dir):
        with open(os.path.join(dir, "config.yaml"), "r") as f:
            config = yaml.load(f)
        if config["embedding_file"] is not None:
            return load_embeddings(config["embedding_file"], typ=config.get("emb_format", None))


def load_embeddings(file, typ='glove', **options):
    """
    Loads either GloVe or word2vec embeddings and wraps it into Embeddings

    Args:
        file: string, path to a file like "GoogleNews-vectors-negative300.bin.gz" or "glove.42B.300d.zip"
        typ: string, either "word2vec", "glove", "fasttext", "mem_map" or "pkl"
        options: dict, other options.
    Returns:
        Embeddings object, wrapper class around Vocabulary embedding matrix.
    """
    type_set = {"word2vec", "glove", "fasttext", "memory_map_dir", "pkl"}
    assert typ.lower() in type_set, "so far only {} foreseen".format(', '.join(type_set))

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

    elif typ.lower() == "memory_map_dir":
        from jack.io.embeddings.memory_map import load_memory_map_dir
        return load_memory_map_dir(file)

    elif typ.lower() == "pkl":
        with open(file, 'rb') as f:
            return pickle.load(f)

    else:
        raise ValueError("Unknown type: {}".format(type))
