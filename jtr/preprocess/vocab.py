# -*- coding: utf-8 -*-
import operator
import sys

import numpy as np
import tensorflow as tf

from jtr.nn.models import get_total_trainable_variables
from jtr.util.tfutil import tfrun


class Vocab(object):
    """
    Vocab objects for use in jtr pipelines.
    """
    DEFAULT_UNK = "<UNK>"

    def __init__(self, unk=DEFAULT_UNK, emb=None, init_from_embeddings=False):
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
        self.emb = emb if emb is not None else lambda _:None #if emb is None: same behavior as for o-o-v words

        if init_from_embeddings and emb is not None:
            self.sym2id = dict(emb.vocabulary.word2idx)
            self.id2sym = {v: k for k, v in emb.vocabulary.word2idx.items()}
            if unk is not None and unk not in self.sym2id:
                self.sym2id[unk] = len(self.sym2id)
                self.id2sym[len(self.id2sym)] = unk
            self.sym2freqs = {w: emb.vocabulary.get_word_count(w) for w in self.sym2id}
            self.frozen = True
            self.next_pos = 0
            self.next_neg = -1*len(self.sym2id)
        else:
            self.sym2id = {}
            # with pos and neg indices
            self.id2sym = {}
            self.next_pos = 0
            self.sym2freqs = {}
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
            sym2id = {sym: self._normalize(id) for sym,id in self.sym2id.items()}
            id2sym = {self._normalize(id): sym for id,sym in self.id2sym.items()}
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
            vec = self.emb(sym)
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
        return id if id >=0 else self.next_pos - id - 1

    def _denormalize(self,id):
        # self.next_pos + i is mapped back to  -1-i
        return id if id < self.next_pos else -1-(id-self.next_pos)

    def get_ids_pretrained(self):
        """return internal or normalized id's (depending on frozen/unfrozen state)
        for symbols that have an embedding in `self.emb` """
        if self.frozen:
            return list(range(self.next_pos,self.next_pos+self.count_pretrained()))
        else:
            return list(range(-1,self.next_neg,-1))

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


class NeuralVocab(Vocab):
    """
    Wrapper around Vocab to go from indices to tensors.
    """

    def __init__(self, base_vocab, embedding_matrix=None,
                 input_size=None, reduced_input_size=None, use_pretrained=True, train_pretrained=False, unit_normalize=True):
        """
        Creates NeuralVocab object from a given Vocab object `base_vocab`.
        Pre-calculates embedding vector (as `Tensor` object) for each symbol in Vocab

        Args:
            `base_vocab`:
            `embedding_matrix`: tensor with shape (len_vocab, input_size). If provided,
              the arguments `input_size`, `use_trained`, `train_pretrained`, and `unit_normalize` are ignored.
            `input_size`: integer; embedding length in case embedding matrix not provided, else ignored.
              If shorter than pre-trained embeddings, only their first `input_size` dimensions are used.
              If longer, extra (Trainable) dimensions are added.
            `reduced_input_size`: integer; optional; ignored in case `None`. If set to positive integer, an additional
              linear layer is introduced to reduce (or extend) the embeddings to the indicated size.
            `use_pretrained`:  boolean; True (default): use pre-trained if available through `base_vocab`.
              False: ignore pre-trained embeddings accessible through `base_vocab`
            `train_pretrained`: boolean; False (default): fix pretrained embeddings. True: continue training.
              Ignored if embedding_matrix is given.
            `unit_normalize`: initialize pre-trained vectors with unit norm
              (note: randomly initialized embeddings are always initialized with expected unit norm)
        """
        super(NeuralVocab, self).__init__(unk=base_vocab.unk, emb=base_vocab.emb)

        assert (embedding_matrix, input_size) is not (None, None), "if no embedding_matrix is provided, define input_size"

        self.freeze() #has no actual functionality here
        base_vocab.freeze() #freeze if not frozen (to ensure fixed non-negative indices)

        self.sym2id = base_vocab.sym2id
        self.id2sym = base_vocab.id2sym
        self.sym2freqs = base_vocab.sym2freqs
        self.unit_normalize = unit_normalize

        if embedding_matrix is None:
            # construct part oov
            n_oov = base_vocab.count_oov()
            n_pre = base_vocab.count_pretrained()
            E_oov = tf.get_variable("embeddings_oov", [n_oov, input_size],
                                     initializer=tf.random_normal_initializer(0, 1./np.sqrt(input_size)),
                                     trainable=True, dtype="float32")
            # stdev = 1/sqrt(length): then expected initial L2 norm is 1

            # construct part pretrained
            if use_pretrained and base_vocab.emb_length is not None:
                # load embeddings into numpy tensor with shape (count_pretrained, min(input_size,emb_length))
                np_E_pre = np.zeros([n_pre, min(input_size, base_vocab.emb_length)]).astype("float32")
                for id in base_vocab.get_ids_pretrained():
                    sym = base_vocab.id2sym[id]
                    i = id - n_oov  #shifted to start from 0
                    np_E_pre[i, :] = base_vocab.emb(sym)[:min(input_size,base_vocab.emb_length)]
                    if unit_normalize:
                        v = np_E_pre[i, :]
                        np_E_pre[i, :] = v / np.sqrt(np.sum(np.square(v)))
                E_pre = tf.get_variable("embeddings_pretrained",
                                        initializer=tf.identity(np_E_pre),
                                        trainable=train_pretrained, dtype="float32")

                if input_size > base_vocab.emb_length:
                    E_pre_ext = tf.get_variable("embeddings_extra", [n_pre, input_size-base_vocab.emb_length],
                                                initializer=tf.random_normal_initializer(0.0, 1. / np.sqrt(base_vocab.emb_length)), dtype="float32", trainable=True)
                    # note: stdev = 1/sqrt(emb_length) means: elements from same normal distr. as normalized first part (in case normally distr.)
                    E_pre = tf.concat([E_pre, E_pre_ext], 1, name="embeddings_pretrained_extended")
            else:
                # initialize all randomly anyway
                E_pre = tf.get_variable("embeddings_not_pretrained", [n_pre, input_size],
                                        initializer=tf.random_normal_initializer(0., 1./np.sqrt(input_size)),
                                        trainable=True, dtype="float32")
                # again: initialize with expected unit norm
            # must be provided is embedding_matrix is None
            self.input_size = input_size
            self.embedding_matrix = tf.concat([E_oov, E_pre], 0, name="embeddings")

        else:
            # ignore input argument input_size
            self.input_size = embedding_matrix.get_shape()[1]
            self.embedding_matrix = embedding_matrix

        if isinstance(reduced_input_size, int) and reduced_input_size > 0:
            # uniform=False for truncated normal
            init = tf.contrib.layers.xavier_initializer(uniform=True)
            self.embedding_matrix = tf.contrib.layers.fully_connected(self.embedding_matrix, reduced_input_size,
                                                                      weights_initializer=init, activation_fn=None)

        # pre-assign embedding vectors to all ids
        # always OK if frozen
        self.id2vec = [tf.nn.embedding_lookup(self.embedding_matrix, idx) for idx in range(len(self))]

    def __call__(self, *args, **kwargs):
        """
        Calling the NeuralVocab object with symbol id's,
        returns a `Tensor` with corresponding embeddings.

        Args:
            `*args`: `Tensor` with integer indices
              (such as a placeholder, to be evaluated when run in a `tf.Session`),
              or list of integer id's,
              or just multiple integer ids as input arguments

        Returns:
            Embedded `Tensor` in case a `Tensor` was provided as input,
            and otherwise a list of embedded input id's under the form of fixed-length embeddings (`Tensor` objects).
        """
        # tuple with length 1: then either list with ids, tensor with ids, or single id
        if len(args) == 1:
            if isinstance(args[0], list):
                ids = args[0]
            elif tf.contrib.framework.is_tensor(args[0]):
                # return embedded tensor
                return tf.nn.embedding_lookup(self.embedding_matrix, args[0])
            else:
                return self.id2vec[args[0]]
        else: # tuple with ids
            ids = args
        return [self.id2vec[id] for id in ids]

    def get_embedding_matrix(self):
        return self.embedding_matrix


if __name__ == '__main__':
    import doctest
    tf.set_random_seed(1337)

    print(doctest.testmod())
