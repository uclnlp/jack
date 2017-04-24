# -*- coding: utf-8 -*-
import operator
import sys

import numpy as np
import tensorflow as tf
import logging

from jtr.nn.models import get_total_trainable_variables
from jtr.util.tfutil import tfrun

logger = logging.getLogger(__name__)


class Vocab(object):
    """
    Vocab objects for use in jtr pipelines.

    Example:

        >>> #Test Vocab without pre-trained embeddings
        >>> vocab = Vocab()
        >>> print(vocab("blah"))
        1
        >>> print(vocab("bluh"))
        2
        >>> print(vocab("bleh"))
        3
        >>> print(vocab("bluh"))
        2
        >>> print(vocab("hello"))
        4
        >>> print(vocab("world"))
        5

        >>> #Sym2id:
        >>> for k in sorted(vocab.sym2id.keys()):
        ...     print(k,' : ',vocab.sym2id[k])
        <UNK>  :  0
        blah  :  1
        bleh  :  3
        bluh  :  2
        hello  :  4
        world  :  5

        >>> #Test Vocab with pre-trained embeddings
        >>> from jtr.load.embeddings.vocabulary import Vocabulary as EVocab
        >>> from jtr.load.embeddings import Embeddings
        >>> np.random.seed(0)
        >>> tf.set_random_seed(0)
        >>> emb = Embeddings(EVocab({'blah':0,'bluh':1,'bleh':2}), np.random.rand(3,4))
        >>> vocab = Vocab(emb=emb)
        >>> print(vocab.id2sym)
        {0: 'blah', 1: 'bluh', 2: 'bleh', 3: '<UNK>'}
        >>> print(vocab("blah"))
        0
        >>> print(vocab("bluh"))
        1
        >>> print(vocab("bleh"))
        2
        >>> print(vocab("bluh"))
        1
        >>> print(vocab("hello"))
        4
        >>> print(vocab("world"))
        5
        >>> print(vocab.n_pretrained)
        3
        >>> vocab.get_ids_pretrained()
        [0, 1, 2]
        >>> vocab.get_ids_oov()
        [3, 4, 5]

        >>> #Test calling frozen Vocab object
        >>> vocab.freeze()
        >>> vocab.sym2id[vocab.unk]
        3
        >>> vocab(['bluh','world','wake','up']) #last 2 are new words, hence unknown
        [1, 5, 3, 3]
        >>> #Test calling unfrozen Vocab object
        >>> vocab.unfreeze()
        >>> encoded_data = vocab(['bluh','world','wake','up', 'world', 'wake','up','wake','up','up','up']) #wake and up are new words, hence added to Vocab
        >>> encoded_data
        [1, 5, 6, 7, 5, 6, 7, 6, 7, 7, 7]
        >>> #before pruning:
        >>> print(sorted(vocab.id2sym.items()))
        [(0, 'blah'), (1, 'bluh'), (2, 'bleh'), (3, '<UNK>'), (4, 'hello'), (5, 'world'), (6, 'wake'), (7, 'up')]
        >>> #frequencies
        >>> print(sorted(vocab.sym2freqs.items()))
        [('<UNK>', 2), ('blah', 1), ('bleh', 1), ('bluh', 3), ('hello', 1), ('up', 5), ('wake', 3), ('world', 3)]
        >>> encoded_data_pruned = vocab.prune(min_freq=2, max_size=4, data=encoded_data)
        >>> #after pruning:
        >>> print(encoded_data_pruned)
        [0, 2, 1, 3, 2, 1, 3, 1, 3, 3, 3]
        >>> print(vocab(['bluh','world','wake','up', 'world', 'wake','up','wake','up','up','up']))
        [0, 2, 1, 3, 2, 1, 3, 1, 3, 3, 3]
        >>> print(sorted(vocab.sym2id.items()))
        [('<UNK>', 1), ('bluh', 0), ('up', 3), ('world', 2)]

    """

    DEFAULT_UNK = "<UNK>"

    def __init__(self, unk=DEFAULT_UNK, emb=None):
        """
        Creates Vocab object.

        Args:
            unk: unknown symbol; not added in case None
            emb: object of type jtr.load.embeddings.embeddings.Embeddings
                 or None (for not using pre-trained embeddings)
        """
        self.unk = unk
        self.emb = emb
        if emb is not None:#start by adding all words with pre-trained embeddings
            self.sym2id = dict(emb.vocabulary.word2idx)
            self.id2sym = {v: k for k, v in emb.vocabulary.word2idx.items()}
            self.n_pretrained = len(self.sym2id)
            if self.unk is not None and self.unk not in self.sym2id:
                self.sym2id[self.unk] = len(self.sym2id)
                self.id2sym[len(self.id2sym)] = self.unk
            self.sym2freqs = {w: 0 for w in self.sym2id}
            self.next = len(self.sym2id)
        else:
            self.n_pretrained = 0
            self.sym2id = {}
            self.id2sym = {}
            self.next = 0
            self.sym2freqs = {}
            if self.unk is not None:
                self.sym2id[self.unk] = 0
                self.id2sym[0] = self.unk
                self.next = 1
                self.sym2freqs[self.unk] = 0

        self.frozen = False

        if emb is not None and hasattr(emb, "lookup") and isinstance(emb.lookup, np.ndarray):
            self.emb_length = emb.lookup.shape[1]
        else:
            self.emb_length = None

    def freeze(self):
        """Freeze current Vocab object (set `self.frozen` to True).
        Further unseen terms will be considered self.unk
        """
        self.frozen = True

    def unfreeze(self):
        """Unfreeze current Vocab object (set `self.frozen` to False).
        Further unseen terms will be added to vocab
        """
        self.frozen = False

    def get_id(self, sym):
        """
        Returns the id of `sym`.
        If self.frozen==False, `sym` is added (or its frequency +1 if already in the vocab), and its id returned.
        If self.frozen==True: if `sym` is a new symbol, the id for `self.unk` (unknown) is returned, if available,
        and otherwise `None` (only possible when input argument `unk` for `Vocab.__init__()` was set to `None`, e.g.,
        for classification labels; it is assumed action is taken in the pipeline
        creating or calling the `Vocab` object, when `None` is encountered).

        Args:
            `sym`: symbol (e.g., token)
        """
        if not self.frozen:
            if sym not in self.sym2id:
                self.sym2id[sym] = self.next
                self.id2sym[self.next] = sym
                self.next += 1
                self.sym2freqs[sym] = 1
            else:
                self.sym2freqs[sym] += 1
            return self.sym2id[sym]
        elif sym in self.sym2id:  #if frozen and sym already in vocab
            return self.sym2id[sym]
        elif self.unk in self.sym2id: #if frozen and sym not in vocab but unk exists in vocab
            self.sym2freqs[self.unk] += 1
            return self.sym2id[self.unk]
        else:  #if self.unk is None
            return None


    def get_sym(self, id):
        """returns symbol for a given id, and None if not found."""
        return None if not id in self.id2sym else self.id2sym[id]

    def __call__(self, *args, **kwargs):
        """
        calls the `get_id` function for the provided symbol(s), which adds symbols to the Vocab if needed and allowed,
        and returns the id(s).

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

    def get_ids_pretrained(self):
        """return ids for symbols that have an embedding in `self.emb` """
        return list(range(self.n_pretrained))

    def get_ids_oov(self):
        """return out-of-vocab id's"""
        return list(range(self.n_pretrained, self.next))

    def get_ids(self):
        return list(range(len(self.sym2id)))

    def get_syms(self):
        return [self.id2sym[id] for id in self.get_ids()]

    def count_pretrained(self):
        """equivalent to `len(get_ids_pretrained())`"""
        return self.n_pretrained

    def count_oov(self):
        """equivalent to `len(get_ids_oov())`"""
        return self.next - self.n_pretrained

    def prune(self, min_freq=1, max_size=None):
        """
        prunes Vocab in place, according to min_freq and max_size.
        After pruning, the Vocab object gets frozen.
        (Unfreezing and adding tokens is possible although not advised: any previously pruned token
        will be added again as out-of-vocab.)

        Args:
            min_freq: int (default 1: filtering out symbols from self.emb that don't appear in the training data)
            max_size: int or None. Resulting Vocab will have at most max_size symbols

        Returns:
            function that allows mapping old id's to id's after this pruning operation

        """
        #filter symbols
        #syms_oov = [self.id2sym[id] for id in self.get_ids_oov()]
        #syms_pretrained = [self.id2sym[id] for id in self.get_ids_pretrained()]
        filtered_syms = [sym for sym in self.get_syms() if self.sym2freqs[sym] >= min_freq]
        filtered_syms = sorted(filtered_syms, key=lambda s: self.sym2freqs[s], reverse=True)

        if isinstance(max_size, int) and len(filtered_syms) >= max_size:
            filtered_syms = filtered_syms[:max_size]
            if self.unk is not None and not self.unk in filtered_syms:
                filtered_syms[-1] = self.unk
        elif self.unk is not None and not self.unk in filtered_syms:
            filtered_syms.append(self.unk)


        #create mapping from old ids to new ids (or the id of self.unk for those left out)
        id_map = [None for _ in range(len(self.sym2id))]
        next_new = 0
        for old_id in self.get_ids_pretrained():
            if self.id2sym[old_id] in filtered_syms:
                id_map[old_id] = next_new
                next_new += 1
        n_pretrained_new = next_new
        for old_id in self.get_ids_oov():
            if self.id2sym[old_id] in filtered_syms:
                id_map[old_id] = next_new
                next_new += 1
        #id_map contains None for filtered out symbols; should become the new id of self.unk
        if self.unk is not None: #then self.unk in filtered_syms
            unk_id_new = id_map[self.sym2id[self.unk]]
            id_map = [unk_id_new if id_new is None else id_new for id_new in id_map]

        #construct id transformation function
        id_trf = lambda old_id: id_map[old_id] if old_id < len(self) else None

        #create updated fields:
        sym2id_new = {sym: id_map[self.sym2id[sym]] for sym in filtered_syms}
        id2sym_new = {id: sym for sym, id in sym2id_new.items()}
        sym2freqs_new = {sym: self.sym2freqs[sym] for sym in filtered_syms}

        #update self:
        self.n_pretrained = n_pretrained_new
        self.sym2id = sym2id_new
        self.id2sym = id2sym_new
        self.sym2freqs = sym2freqs_new
        self.next = next_new

        #freeze after pruning
        self.freeze()

        return id_trf

    @staticmethod
    def vocab_to_tensor(vocab, emb_length=None, train_pretrained=False,
                        name='embedding_tensor'):
        """
        Constructs embedding tensor from vocab. Embeddings are initialized with (expected) unit norm.
        Args:
            vocab: populated Vocab
            emb_length: length of embeddings; ignored if vocab.emb_length is not None
            train_pretrained: continue training pre-trained embeddings
        Returns:
            embedding tensor
        """
        assert len(vocab) > 0, "vocab_to_tensor requires a populated Vocab object"

        np_normalize = lambda v: v / np.sqrt(np.sum(np.square(v)))
        emb_length = emb_length if vocab.emb_length is None else vocab.emb_length
        n_oov, n_pre = vocab.count_oov(), vocab.count_pretrained()

        if n_oov > 0:
            E_oov = tf.get_variable("embeddings_oov", [n_oov, emb_length],
                                    initializer=tf.random_normal_initializer(0, 1. / np.sqrt(emb_length)),
                                    trainable=True, dtype="float32")
        if n_pre > 0:
            np_E_pre = np.zeros([n_pre, emb_length]).astype("float32")
            for i in vocab.get_ids_pretrained():
                sym = vocab.id2sym[i]
                np_E_pre[i, :] = np_normalize(vocab.emb(sym))
            E_pre = tf.get_variable("embeddings_pretrained", initializer=tf.identity(np_E_pre),
                                    trainable=train_pretrained, dtype="float32")

        logger.debug('Created embedding tensor with %d out-of-vocab and %d pre-trained embeddings'%
                     (n_oov, n_pre))

        if n_oov > 0 and n_pre > 0:
            return tf.concat([E_oov, E_pre], 0, name=name)
        elif n_pre == 0:
            return tf.identity(E_oov, name=name)
        else:
            return tf.identity(E_pre, name=name)


# class NeuralVocab(Vocab):
#     """
#     Wrapper around Vocab to go from indices to tensors.
#
#     Example:
#         >>> #Start from same Vocab as the doctest example in Vocab
#         >>> def emb(w):
#         ...    v = {'blah':[1.7,0,.3],'bluh':[0,1.5,0.5],'bleh':[0,0,2]}
#         ...    return None if not w in v else v[w]
#         >>> vocab = Vocab(emb=emb)
#         >>> vocab("blah", "bluh", "bleh", "hello", "world")  #symbols as multiple arguments
#         [-1, -2, -3, 1, 2]
#         >>> vocab(['bluh','world','wake','up']) #as list of symbols
#         [-2, 2, 3, 4]
#
#         >>> #Create NeuralVocab object
#         >>> with tf.variable_scope('neural_test1'):
#         ...     nvocab = NeuralVocab(vocab, None, 3, unit_normalize=True)
#         ...     tfrun(nvocab(vocab("world")))
#         array([ 0.46077079,  0.38316524, -0.63771147], dtype=float32)
#         >>> tra1 = get_total_trainable_variables()
#
#
#         >>> #Test NeuralVocab with pre-trained embeddings  (case: input_size larger than pre-trained embeddings)
#         >>> with tf.variable_scope('neural_test2'):
#         ...     for w in ['blah','bluh','bleh']:
#         ...         w, emb(w)
#         ...     nvocab = NeuralVocab(vocab, None, 4, unit_normalize=True, use_pretrained=True, train_pretrained=False)
#         ...     tfrun(nvocab.embedding_matrix)
#         ('blah', [1.7, 0, 0.3])
#         ('bluh', [0, 1.5, 0.5])
#         ('bleh', [0, 0, 2])
#         array([[-0.26461828,  0.65265107,  0.39575091, -0.30496973],
#                [ 0.48515028,  0.19880073, -0.02314733, -0.02336031],
#                [ 0.26688093, -0.24634691,  0.2248017 ,  0.24709973],
#                [-0.39200979, -0.49848005, -1.11226082, -0.15154324],
#                [ 0.46785676,  1.64755058,  0.15274598,  0.17200644],
#                [ 0.98478359,  0.        ,  0.17378533, -0.46795556],
#                [ 0.        ,  0.94868326,  0.31622776, -0.72465843],
#                [ 0.        ,  0.        ,  1.        , -0.46098801]], dtype=float32)
#         >>> get_total_trainable_variables()-tra1
#         23
#
#     Interpretation of number of trainable variables from neural_test2:
#     out-of-vocab: 8 - 3 = 5 symbols, with each 4 dimensions = 20;
#     for fixed pre-trained embeddings with length 3, three times 1 extra trainable dimension for total embedding length 4.
#     Total is 23.
#     """
#
#     def __init__(self, base_vocab, embedding_matrix=None,
#                  input_size=None, reduced_input_size=None, use_pretrained=True, train_pretrained=False, unit_normalize=True):
#         """
#         Creates NeuralVocab object from a given Vocab object `base_vocab`.
#         Pre-calculates embedding vector (as `Tensor` object) for each symbol in Vocab
#
#         Args:
#             `base_vocab`:
#             `embedding_matrix`: tensor with shape (len_vocab, input_size). If provided,
#               the arguments `input_size`, `use_trained`, `train_pretrained`, and `unit_normalize` are ignored.
#             `input_size`: integer; embedding length in case embedding matrix not provided, else ignored.
#               If shorter than pre-trained embeddings, only their first `input_size` dimensions are used.
#               If longer, extra (Trainable) dimensions are added.
#             `reduced_input_size`: integer; optional; ignored in case `None`. If set to positive integer, an additional
#               linear layer is introduced to reduce (or extend) the embeddings to the indicated size.
#             `use_pretrained`:  boolean; True (default): use pre-trained if available through `base_vocab`.
#               False: ignore pre-trained embeddings accessible through `base_vocab`
#             `train_pretrained`: boolean; False (default): fix pretrained embeddings. True: continue training.
#               Ignored if embedding_matrix is given.
#             `unit_normalize`: initialize pre-trained vectors with unit norm
#               (note: randomly initialized embeddings are always initialized with expected unit norm)
#         """
#         super(NeuralVocab, self).__init__(unk=base_vocab.unk, emb=base_vocab.emb)
#
#         assert (embedding_matrix, input_size) is not (None, None), "if no embedding_matrix is provided, define input_size"
#
#         self.freeze() #has no actual functionality here
#         base_vocab.freeze() #freeze if not frozen (to ensure fixed non-negative indices)
#
#         self.sym2id = base_vocab.sym2id
#         self.id2sym = base_vocab.id2sym
#         self.sym2freqs = base_vocab.sym2freqs
#         self.unit_normalize = unit_normalize
#
#         def np_normalize(v):
#             return v / np.sqrt(np.sum(np.square(v)))
#
#         if embedding_matrix is None:
#             # construct part oov
#             n_oov = base_vocab.count_oov()
#             n_pre = base_vocab.count_pretrained()
#             E_oov = tf.get_variable("embeddings_oov", [n_oov, input_size],
#                                      initializer=tf.random_normal_initializer(0, 1./np.sqrt(input_size)),
#                                      trainable=True, dtype="float32")
#             # stdev = 1/sqrt(length): then expected initial L2 norm is 1
#
#             # construct part pretrained
#             if use_pretrained and base_vocab.emb_length is not None:
#                 # load embeddings into numpy tensor with shape (count_pretrained, min(input_size,emb_length))
#                 np_E_pre = np.zeros([n_pre, min(input_size, base_vocab.emb_length)]).astype("float32")
#                 for id in base_vocab.get_ids_pretrained():
#                     sym = base_vocab.id2sym[id]
#                     i = id - n_oov  #shifted to start from 0
#                     np_E_pre[i, :] = base_vocab.emb(sym)[:min(input_size,base_vocab.emb_length)]
#                     if unit_normalize:
#                         np_E_pre[i, :] = np_normalize(np_E_pre[i, :])
#                 E_pre = tf.get_variable("embeddings_pretrained",
#                                         initializer=tf.identity(np_E_pre),
#                                         trainable=train_pretrained, dtype="float32")
#
#                 if input_size > base_vocab.emb_length:
#                     E_pre_ext = tf.get_variable("embeddings_extra", [n_pre, input_size-base_vocab.emb_length],
#                                                 initializer=tf.random_normal_initializer(0.0, 1. / np.sqrt(base_vocab.emb_length)), dtype="float32", trainable=True)
#                     # note: stdev = 1/sqrt(emb_length) means: elements from same normal distr. as normalized first part (in case normally distr.)
#                     E_pre = tf.concat([E_pre, E_pre_ext],
#                             1, name="embeddings_pretrained_extended")
#             else:
#                 # initialize all randomly anyway
#                 E_pre = tf.get_variable("embeddings_not_pretrained", [n_pre, input_size],
#                                         initializer=tf.random_normal_initializer(0., 1./np.sqrt(input_size)),
#                                         trainable=True, dtype="float32")
#                 # again: initialize with expected unit norm
#
#             # must be provided is embedding_matrix is None
#             self.input_size = input_size
#             self.embedding_matrix = tf.concat([E_oov, E_pre],
#                     0, name="embeddings")
#
#         else:
#             # ignore input argument input_size
#             self.input_size = embedding_matrix.get_shape()[1]
#             self.embedding_matrix = embedding_matrix
#
#         if isinstance(reduced_input_size, int) and reduced_input_size > 0:
#             # uniform=False for truncated normal
#             init = tf.contrib.layers.xavier_initializer(uniform=True)
#             self.embedding_matrix = tf.contrib.layers.fully_connected(self.embedding_matrix, reduced_input_size,
#                                                                       weights_initializer=init, activation_fn=None)
#
#         # pre-assign embedding vectors to all ids
#         # always OK if frozen
#         self.id2vec = [tf.nn.embedding_lookup(self.embedding_matrix, idx) for idx in range(len(self))]
#
#     def embed_symbol(self, ids):
#         """returns embedded id's
#
#         Args:
#             `ids`: integer, ndarray with np.int32 integers, or tensor with tf.int32 integers.
#             These integers correspond to (normalized) id's for symbols in `self.base_vocab`.
#
#         Returns:
#             tensor with id's embedded by numerical vectors (in last dimension)
#         """
#         return tf.nn.embedding_lookup(self.embedding_matrix, ids)
#
#     def __call__(self, *args, **kwargs):
#         """
#         Calling the NeuralVocab object with symbol id's,
#         returns a `Tensor` with corresponding embeddings.
#
#         Args:
#             `*args`: `Tensor` with integer indices
#               (such as a placeholder, to be evaluated when run in a `tf.Session`),
#               or list of integer id's,
#               or just multiple integer ids as input arguments
#
#         Returns:
#             Embedded `Tensor` in case a `Tensor` was provided as input,
#             and otherwise a list of embedded input id's under the form of fixed-length embeddings (`Tensor` objects).
#         """
#         # tuple with length 1: then either list with ids, tensor with ids, or single id
#         if len(args) == 1:
#             if isinstance(args[0], list):
#                 ids = args[0]
#             elif tf.contrib.framework.is_tensor(args[0]):
#                 # return embedded tensor
#                 return self.embed_symbol(args[0])
#             else:
#                 return self.id2vec[args[0]]
#         else: # tuple with ids
#             ids = args
#         return [self.id2vec[id] for id in ids]
#
#     def get_embedding_matrix(self):
#         return self.embedding_matrix


if __name__ == '__main__':
    import doctest
    tf.set_random_seed(1337)

    print(doctest.testmod())
