# -*- coding: utf-8 -*-

import operator
import pickle
import sys
from collections import OrderedDict


class Vocab:
    """
    Vocab objects for use in jack pipelines.
    """
    DEFAULT_UNK = "<UNK>"

    def __init__(self, unk=DEFAULT_UNK, vocab: dict = None):
        """
        Creates Vocab object.

        Args:
            `unk`: symbol for unknown term (default: "<UNK>").
              If set to `None`, and `None` is not included as symbol while unfrozen,
              it will return `None` upon calling `get_id(None)` when frozen.
            `vocab`: init from vocab dict (sym -> id)
        """
        self._unk = unk
        if vocab is not None:
            self._sym2id = dict(vocab)
            self._id2sym = {v: k for k, v in vocab.items()}
            if unk is not None and unk not in self._sym2id:
                self._sym2id[unk] = len(self._sym2id)
                self._id2sym[len(self._id2sym)] = unk
            self._sym2freqs = {w: 0 for w in self._sym2id}
            self._frozen = True
        else:
            self._sym2id = {}
            # with pos and neg indices
            self._id2sym = {}
            self._sym2freqs = OrderedDict()
            if unk is not None:
                self._sym2id[unk] = 0
                # with pos and neg indices
                self._id2sym[0] = unk
                self._sym2freqs[unk] = 0
            self._frozen = False

    def freeze(self):
        """Freeze current Vocab object (set `self._frozen` to True)."""
        # if any pretrained have been encountered
        self._frozen = True

    def unfreeze(self):
        """Unfreeze current Vocab object (set `self.frozen` to False)."""
        self._frozen = False

    def get_id(self, sym):
        """Get id of symbol. Counts frequency if not frozen."""
        if not self._frozen:
            if sym not in self._sym2id:
                self._sym2id[sym] = len(self._sym2id)
                self._id2sym[len(self._id2sym)] = sym
                self._sym2freqs[sym] = 1
            else:
                self._sym2freqs[sym] += 1
        return self._sym2id.get(sym, self._sym2id.get(self._unk))

    def get_sym(self, id):
        """returns symbol for a given id (consistent with the `self.frozen` state), and None if not found."""
        return self._id2sym.get(id)

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
        return len(self._id2sym)

    def __contains__(self, sym):
        """checks if `sym` already in the Vocab object"""
        return sym in self._sym2id

    @property
    def frozen(self):
        return self._frozen

    @property
    def unk(self):
        return self._unk

    def prune(self, min_freq=5, max_size=sys.maxsize):
        """returns new Vocab object, pruned based on minimum symbol frequency"""
        pruned_vocab = Vocab(unk=self._unk)
        cnt = 0
        for sym, freq in sorted(self._sym2freqs.items(), key=operator.itemgetter(1), reverse=True):
            # for sym in self.sym2freqs:
            # freq = self.sym2freqs[sym]
            cnt += 1
            if freq >= min_freq and cnt < max_size:
                pruned_vocab(sym)
                pruned_vocab._sym2freqs[sym] = freq
        if self._frozen:
            # if original Vocab was frozen, freeze new one
            pruned_vocab.freeze()

        return pruned_vocab

    def store(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        # backwards compability
        new_data = {}
        for k in data:
            if k not in self.__dict__:
                new_k = '_' + k
                if new_k in self.__dict__:
                    new_data[new_k] = data[k]
            else:
                new_data[k] = data[k]
        self.__dict__ = new_data
