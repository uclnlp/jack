# -*- coding: utf-8 -*-

import operator
import sys

from collections import OrderedDict

import numpy as np
import os
import pickle

from sacred.optional import yaml

from jack.io.embeddings import Embeddings, load_embeddings


class Vocab:
    """
    Vocab objects for use in jack pipelines.
    """
    DEFAULT_UNK = "<UNK>"

    def __init__(self, unk=DEFAULT_UNK, emb: Embeddings = None, init_from_embeddings=False):
        """
        Creates Vocab object.

        Args:
            `unk`: symbol for unknown term (default: "<UNK>").
              If set to `None`, and `None` is not included as symbol while unfrozen,
              it will return `None` upon calling `get_id(None)` when frozen.
            `emb`: function handle; returns pre-trained embedding (fixed-size numerical list or ndarray)
              for a given symbol, and None for unknown symbols.
        """
        self.next_pos = 0
        self.next_neg = -1
        self.unk = unk
        self.emb = emb  # if emb is not None else lambda _:None #if emb is None: same behavior as for o-o-v words

        if init_from_embeddings and emb is not None:
            self.sym2id = dict(emb.vocabulary)
            self.id2sym = {v: k for k, v in emb.vocabulary.items()}
            if unk is not None and unk not in self.sym2id:
                self.sym2id[unk] = len(self.sym2id)
                self.id2sym[len(self.id2sym)] = unk
            self.sym2freqs = {w: None for w in self.sym2id}
            self.frozen = True
            self.next_pos = 0
            self.next_neg = -1 * len(self.sym2id)
        else:
            self.sym2id = {}
            # with pos and neg indices
            self.id2sym = {}
            self.next_pos = 0
            self.sym2freqs = OrderedDict()
            if unk is not None:
                self.sym2id[unk] = 0
                # with pos and neg indices
                self.id2sym[0] = unk
                self.next_pos = 1
                self.sym2freqs[unk] = 0
            self.frozen = False

        if emb is not None and hasattr(emb, "lookup") and isinstance(emb.lookup, np.ndarray):
            self.emb_length = emb.lookup.shape[1]
        else:
            self.emb_length = None

    def _get_emb(self, word):
        return self.emb(word) if self.emb is not None else None

    def freeze(self):
        """Freeze current Vocab object (set `self.frozen` to True).
        To be used after loading symbols from a given corpus;
        transforms all internal symbol id's to positive indices (for use in tensors).

        - additional calls to the __call__ method will return the id for the unknown symbold
        - out-of-vocab id's are positive integers and do not change
        - id's of symbols with pre-trained embeddings are converted to positive integer id's,
          counting up from the all out-of-vocab id's.
        """
        # if any pretrained have been encountered
        if not self.frozen and self.next_neg < -1:
            sym2id = {sym: self._normalize(id) for sym, id in self.sym2id.items()}
            id2sym = {self._normalize(id): sym for id, sym in self.id2sym.items()}
            self.sym2id = sym2id
            self.id2sym = id2sym
        self.frozen = True

    def unfreeze(self):
        """Unfreeze current Vocab object (set `self.frozen` to False).
        Caution: use with care! Unfreezing a Vocab, adding new terms, and again Freezing it,
        will result in shifted id's for pre-trained symbols.

        - maps all normalized id's to the original internal id's.
        - additional calls to __call__ will allow adding new symbols to the vocabulary.
        """
        if self.frozen and self.next_neg < -1:
            sym2id = {sym: self._denormalize(id) for sym, id in self.sym2id.items()}
            id2sym = {self._denormalize(id): sym for id, sym in self.id2sym.items()}
            self.sym2id = sym2id
            self.id2sym = id2sym
        self.frozen = False

    def get_id(self, sym):
        """
        Returns the id of `sym`; different behavior depending on the state of the Vocab:

        - In case self.frozen==False (default): returns internal id,
          that is, positive for out-of-vocab symbol, negative for symbol
          found in `self.emb`. If `sym` is a new symbol, it is added to the Vocab.

        - In case self.frozen==True (after explicit call to 'freeze()', or after building a `NeuralVocab` with it):
          Returns normalized id (positive integer, also for symbols with pre-trained embedding)
          If `sym` is a new symbol, the id for unknown terms is returned, if available,
          and otherwise `None` (only possible when input argument `unk` for `Vocab.__init__()` was set to `None`, e.g. ;
          for classification labels; it is assumed action is taken in the pipeline
          creating or calling the `Vocab` object, when `None` is encountered).

        Args:
            `sym`: symbol (e.g., token)
        """
        if not self.frozen:
            vec = self._get_emb(sym)
            if self.emb_length is None and vec is not None:
                self.emb_length = len(vec) if isinstance(vec, list) else vec.shape[0]
            if sym not in self.sym2id:
                if vec is None:
                    self.sym2id[sym] = self.next_pos
                    self.id2sym[self.next_pos] = sym
                    self.next_pos += 1
                else:
                    self.sym2id[sym] = self.next_neg
                    self.id2sym[self.next_neg] = sym
                    self.next_neg -= 1
                self.sym2freqs[sym] = 1
            else:
                self.sym2freqs[sym] += 1
        if sym in self.sym2id:
            return self.sym2id[sym]
        else:
            if self.unk in self.sym2id:
                return self.sym2id[self.unk]
            # can happen for `Vocab` initialized with `unk` argument set to `None`
            else:
                return None

    def get_sym(self, id):
        """returns symbol for a given id (consistent with the `self.frozen` state), and None if not found."""
        return None if not id in self.id2sym else self.id2sym[id]

    def __call__(self, *args, **kwargs):
        """
        calls the `get_id` function for the provided symbol(s), which adds symbols to the Vocab if needed and allowed,
        and returns their id(s).

        Args:
            *args: a single symbol, a list of symbols, or multiple symbols
        """
        symbols = args
        if len(args) == 1:
            if isinstance(args[0], list):
                symbols = args[0]
            else:
                return self.get_id(args[0])
        return [self.get_id(sym) for sym in symbols]

    def __len__(self):
        """returns number of unique symbols (including the unknown symbol)"""
        return len(self.id2sym)

    def __contains__(self, sym):
        """checks if `sym` already in the Vocab object"""
        return sym in self.sym2id

    def _normalize(self, id):
        """map original (pos/neg) ids to normalized (non-neg) ids: first new symbols, then those in emb"""
        # e.g. -1 should be mapped to self.next_pos + 0
        # e.g. -3 should be mapped to self.next_pos + 2
        return id if id >= 0 else self.next_pos - id - 1

    def _denormalize(self, id):
        # self.next_pos + i is mapped back to  -1-i
        return id if id < self.next_pos else - 1 - (id - self.next_pos)

    def get_ids_pretrained(self):
        """return internal or normalized id's (depending on frozen/unfrozen state)
        for symbols that have an embedding in `self.emb` """
        if self.frozen:
            return list(range(self.next_pos, self.next_pos + self.count_pretrained()))
        else:
            return list(range(-1, self.next_neg, -1))

    def get_ids_oov(self):
        """return out-of-vocab id's (indep. of frozen/unfrozen state)"""
        return list(range(self.next_pos))

    def count_pretrained(self):
        """equivalent to `len(get_ids_pretrained())`"""
        return -self.next_neg - 1

    def count_oov(self):
        """equivalent to `len(get_ids_oov())`"""
        return self.next_pos

    def prune(self, min_freq=5, max_size=sys.maxsize):
        """returns new Vocab object, pruned based on minimum symbol frequency"""
        pruned_vocab = Vocab(unk=self.unk, emb=self.emb)
        cnt = 0
        for sym, freq in sorted(self.sym2freqs.items(), key=operator.itemgetter(1), reverse=True):
            # for sym in self.sym2freqs:
            # freq = self.sym2freqs[sym]
            cnt += 1
            if freq >= min_freq and cnt < max_size:
                pruned_vocab(sym)
                pruned_vocab.sym2freqs[sym] = freq
        if self.frozen:
            # if original Vocab was frozen, freeze new one
            pruned_vocab.freeze()

        return pruned_vocab

    def store(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        conf_file = path + "_conf.yaml"
        emb_file = path + "_emb.pkl"
        remainder_file = path + ".pkl"
        with open(conf_file, "w") as f:
            yaml.dump({"embedding_file": self.emb.filename, "emb_format": self.emb.emb_format}, f)

        if self.emb.filename is None:
            with open(emb_file, "wb") as f:
                pickle.dump(self.emb, f)
        remaining = {k: self.__dict__[k] for k in self.__dict__ if k != "emb"}
        with open(remainder_file, "wb") as f:
            pickle.dump(remaining, f)

    def load(self, path: str):
        conf_file = path + "_conf.yaml"
        emb_file = path + "_emb.pkl"
        remainder_file = path + ".pkl"
        with open(conf_file, "r") as f:
            config = yaml.load(f)

        if config["embedding_file"] is not None:
            emb = load_embeddings(config["embedding_file"], typ=config.get("emb_format", None))
        else:
            with open(emb_file, "rb") as f:
                emb = pickle.load(f)

        with open(remainder_file, "rb") as f:
            remaining = pickle.load(f)

        self.__dict__ = remaining
        self.__dict__["emb"] = emb
