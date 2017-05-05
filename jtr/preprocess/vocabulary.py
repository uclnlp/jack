import numpy as np
import sys
import tensorflow as tf

import logging

logger = logging.getLogger(__name__)



class Vocab(object):
    """
    Vocab object, with main characteristics:
        - allows adding (nested lists of) symbols
        - can be pruned based on max_size or min_freq
        - allows encoding (nested lists of) symbols
        - can make use of pre-trained embeddings (only reads those present in the corpus)
        - at test time, able to add additional symbols if available as pre-trained embeddings
        - creates corresponding tensor; at test time able to extend it if extra symbols detected
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
        #todo: simplify for any emb that maps symbol to embedding vector

        self.unk = unk

        self.symset_oov = set() if unk is None else {unk}
        self.symset_pt = set() #symbols of encountered pre-trained symbols
        self.sym2freqs = dict() if unk is None else {unk: 0} #keep track of encountered symbols before building

        if emb is not None and hasattr(emb, "lookup") and isinstance(emb.lookup, np.ndarray):
            self.emb = emb
            self.emb_length = emb.lookup.shape[1]
        else:
            self.emb, self.emb_length = None, None

        self.built = False
        self.sym2id = dict() if unk is None else {unk: 0}
        self.id2sym = []


    def add_pretrained_for_testing(self, *data):
        self.add(*data, allow_oov=False, allow_pt=True, count_freqs=False)


    def add(self, *data, allow_oov=True, allow_pt=True, count_freqs=True):
        """
        adds (nested sequence(s) of) symbols to Vocab object.
        Only possible before calling build().

        Args:
            *data: (nested sequence(s) of) symbols
            allow_oov: allow adding symbols for which no pre-trained embedding is available
            allow_pt: allow adding symbols for which a pre-trained embedding is available
            count_freqs: add to symbol frequency counts
        """
        assert not self.built, "adding data to Vocab object no longer allowed after calling Vocab.build()"
        if len(data) > 1:
            for sym in data:
                self.add(sym, allow_oov=allow_oov, allow_pt=allow_pt, count_freqs=count_freqs)
        elif len(data) == 1:
            if isinstance(data[0], list) or isinstance(data[0], tuple):
                for sym in data[0]:
                    self.add(sym, allow_oov=allow_oov, allow_pt=allow_pt, count_freqs=count_freqs)
            else: #add single symbol
                sym = data[0]
                # check if already seen (only add to frequency counts if allowed)
                if sym in self:
                    if count_freqs and ((sym in self.symset_oov and allow_oov) or (sym in self.symset_pt and allow_pt)):
                        self.sym2freqs[sym] += 1
                # for new terms: check if pre-trained
                elif self._has_pt(sym):
                    if allow_pt:
                        self.symset_pt.add(sym)
                        self.sym2freqs[sym] = 1 if count_freqs else 0
                # or add as out-of-vocab
                elif allow_oov:
                    self.symset_oov.add(sym)
                    self.sym2freqs[sym] = 1 if count_freqs else 0
                # else: unknown
                elif self.emb is not None and count_freqs:
                    self.sym2freqs[self.unk] += 1


    def build(self, min_freq=0, max_size=sys.maxsize):
        """
        prepare Vocab object for encoding data:
        (1) prune if min_freq > 1 or max_size < vocab size (unknown symbol not included)
        (2) build mapping from symbols to indices
        If fix_pretrained==False: allow automatic extension of Vocab during encoding,
        when new symbols are seen for which pre-trained embeddings are available.
        """
        #(1) prune Vocab object based on min_freq and/or max_size
        needs_pruning = min_freq > 1 or len(self) > max_size
        if needs_pruning:
            #prune on min_freq
            filtered_syms = [sym for sym in self.sym2freqs if self.sym2freqs[sym] >= min_freq]
            #prune on max_size (sort on freq but also symbol, for deterministic pruning)
            if len(filtered_syms) > max_size:
                filtered_syms = sorted(filtered_syms, key=lambda s: (-self.sym2freqs[s], s))[:max_size]
            #restore self.unk if need be
            if self.unk is not None and not self.unk in filtered_syms:
                filtered_syms.append(self.unk)
            #prune Vocab
            filtered_syms = set(filtered_syms)
            self.symset_oov = filtered_syms.intersection(self.symset_oov)
            self.symset_pt = filtered_syms.intersection(self.symset_pt)
            sym2freqs_new = {sym: self.sym2freqs[sym] for sym in self.sym2freqs if sym in filtered_syms}
            if self.unk in sym2freqs_new:
                n_removed = np.sum([self.sym2freqs[sym] for sym in self.sym2freqs if not sym in filtered_syms])
                sym2freqs_new[self.unk] += n_removed
            self.sym2freqs = sym2freqs_new

        #create mapping from syms to ids (sorting: to keep deterministic for reproducibility)
        syms_pt = sorted(list(self.symset_pt))
        syms_oov = sorted(list(self.symset_oov))
        if self.unk in self.symset_oov: #make sure self.unk has index 0
            syms_oov.remove(self.unk)
            syms_oov = [self.unk] + syms_oov

        self.id2sym = syms_oov + syms_pt
        self.sym2id = {sym: i for i, sym in enumerate(self.id2sym)}
        self.built = True
        logger.debug('Vocab object was built with {} out-of-vocab symbols, '\
                     'and {} symbols with pre-trained embeddings'\
                     ''.format(len(self.symset_oov), len(self.symset_pt)))


    def encode(self, *data, keys=[]):
        """
        Encode symbols into ids.
        Only possible after calling build()
        Args:
            *data: (nested) sequence of symbols, or dictionary
            keys:  if a data dictionary is provided: keys for which values need to be encoded (in place)
        Returns: encoded data
        """
        assert self.built, "first build mapping from symbols to ids, using Vocab.build()"
        if len(data) == 1 and isinstance(data[0], dict):
            encoded_data = data[0]
            for key in encoded_data:
                if key in keys:
                    encoded_data[key] = self._code(self._encode_symbol, encoded_data[key])
            return encoded_data
        else:
            return self._code(self._encode_symbol, *data)


    def decode(self, *data, keys=[]):
        """
        Decode ids into symbols.
        Only possible after calling build()
        Args:
            *data: (nested) sequence of symbols, or dictionary
            keys:  if a data dictionary is provided: keys for which values need to be decoded (in place)
        Returns: decoded data
        """
        assert self.built, "first build mapping from symbols to ids, using Vocab.build()"
        if len(data) == 1 and isinstance(data[0], dict):
            decoded_data = data[0]
            for key in decoded_data:
                if key in keys:
                    decoded_data[key] = self._code(self._decode_id, decoded_data[key])
            return decoded_data
        else:
            return self._code(self._decode_id, *data)


    def _code(self, code_fun, *data):
        """
        recursive encoding or decoding of (nested sequence(s) of) symbols, respectively, id's.
        If `data` is a single (nested) sequence:
        returns a corresponding nested sequence.
        If several `data` input arguments are given:
        a tuple with (nested) sequences is returned.
        """
        if len(data) > 1:
            return tuple(self._code(code_fun, d) for d in data)
        elif len(data) == 1:
            if isinstance(data[0], list):
                return [self._code(code_fun, sym) for sym in data[0]]
            elif isinstance(data[0], tuple):
                return tuple(self._code(code_fun, sym) for sym in data[0])
            else:
                return code_fun(data[0])


    def _encode_symbol(self, sym):
        """
        Return id for symbol `sym`. Should only be called after calling self.build().
        Args:
            sym: symbol
        """
        return self.sym2id.get(sym, self.sym2id.get(self.unk, None))


    def _decode_id(self, id):
        return None if id >= len(self.id2sym) else self.id2sym[id]


    def _has_pt(self, sym):
        return self.emb is not None and sym in self.emb.vocabulary.word2idx


    def get_stats(self):
        return {
            'status': 'built' if self.built else 'not built yet',
            'out-of-vocab': len(self.symset_oov),
            'pre-trained': len(self.symset_pt)
        }


    def __contains__(self, sym):
        return sym in self.symset_pt or sym in self.symset_oov


    def __len__(self):
        return len(self.symset_oov) + len(self.symset_pt)


    @staticmethod
    def get_tensor(vocab, emb_length=None, normalize=False, init='uniform', name='embeddings'):
        """
        Creates tensor with dictionary and embeddings.

        Args:
            vocab: object of class Vocab, after being built (vocab.built=True)
            emb_length: ignored unless Vocab.emb = None
            normalize: whether to scale (initial) embedding vectors to have (expected) L2 unit norm
            init: initialization of out-of-vocab embeddings:
                  'uniform': standard uniform distr (or scaled)
                  or 'normal': standard normal distr (or scaled)
            name: name for the resulting tensor
        """
        assert vocab.built, "first build mapping from symbols to ids, using Vocab.build()"
        # if not vocab.built:
        #     vocab.build() #build without pruning

        np_normalize = lambda v: v / np.sqrt(np.sum(np.square(v))) if normalize else v

        k = vocab.emb_length or emb_length
        assert isinstance(k, int) and k > 0, "Initialize `vocab` with pre-trained embeddings, " \
                                             "or provide `emb_length` for `get_tensor`()"
        n_oov, n_pt = len(vocab.symset_oov), len(vocab.symset_pt)

        if n_oov > 0:
            if init == 'normal':
                init_oov = tf.random_normal([n_oov, k], mean=0, stddev=1. / np.sqrt(k) if normalize else 1.)
            else:
                init_oov = tf.random_uniform([n_oov, k], 0, np.sqrt(3. / k) if normalize else 1., dtype=tf.float32)

            E_oov = tf.Variable(initial_value=init_oov, trainable=True,
                                name=name + '_oov', dtype="float32")

        if n_pt > 0:
            np_E_pt = np.zeros([n_pt, k]).astype("float32")
            for sym in vocab.symset_pt:
                i = vocab.sym2id[sym] - n_oov
                np_E_pt[i, :] = np_normalize(vocab.emb(sym))
            E_pt = tf.Variable(initial_value=np_E_pt, trainable=False,
                               name=name + '_pt', dtype="float32")

        if n_oov > 0 and n_pt > 0:
            embedding_tensor = tf.concat([E_oov, E_pt], 0, name=name)
        elif n_pt == 0:
            embedding_tensor = tf.identity(E_oov, name=name)
        else:
            embedding_tensor = tf.identity(E_pt, name=name)
        logger.debug('Embedding matrix with shape {} was constructed'.format(embedding_tensor.get_shape()))

        return embedding_tensor


if __name__=="__main__":
    from jtr.load.embeddings.vocabulary import Vocabulary as EVocab
    from jtr.load.embeddings import Embeddings
    from copy import copy

    corpus = "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae ; " \
             "alteram , Aquitani ; tertiam qui ipsorum lingua Celtae, nostra Galli appellantur . " \
             "Horum omnium fortissimi sunt Belgae , propterea quod a cultu atque humanitate provinciae " \
             "longissime absunt , minimeque ad eos mercatores saepe commeant , " \
             "atque ea quae ad effeminandos animos pertinent important ."
    corpus = corpus.split()

    emb = Embeddings(EVocab({'Horum': 0, 'omnium': 1, 'fortissimi': 2, 'sunt': 3, 'Belgae': 4}), np.random.rand(5, 4))

    #test with allow_oov=False
    vocab = Vocab(emb=emb)
    vocab.add_pretrained_for_testing(corpus)
    for k, v in vocab.get_stats().items():
        print('%s : %s'%(k, str(v)))
    print('sym2freqs after add_pretrained_for_testing:', vocab.sym2freqs)
    vocab.add(corpus, allow_oov=False)
    vocab.build()
    print('sym2id: ', vocab.sym2id)
    print('sym2freqs: ', vocab.sym2freqs)
    t = Vocab.get_tensor(vocab)
    print('emb_tensor has shape ', t.get_shape())
    print('\n\n')

    #test with allow_oov=True
    vocab = Vocab(emb=emb)
    vocab.add(corpus)
    print('symset oov: ', vocab.symset_oov)
    print('symset pt: ', vocab.symset_pt)
    print('Before building:')
    for k, v in vocab.get_stats().items():
        print('{} : {}'.format(k, v))
    print('After building')
    vocab.build(max_size=25)
    for k, v in vocab.get_stats().items():
        print('{} : {}'.format(k, v))
    print('pretrained symbols: ', vocab.symset_pt)
    print('vocab.id2sym : ', vocab.id2sym)
    print('vocab.sym2id : ', vocab.sym2id)
    print('vocab.sym2freqs : ', vocab.sym2freqs)
    t = Vocab.get_tensor(vocab)
    print('emb_tensor has shape ', t.get_shape())

    set1_pt = copy(vocab.symset_pt)
    corpus_ids = vocab.encode(corpus)
    set2_pt = copy(vocab.symset_pt)
    print('encoded corpus: ', corpus_ids)
    print('during encoding, added previsouly filtered pretrained symbols: ', set2_pt.difference(set1_pt))
    t = Vocab.get_tensor(vocab)
    print('emb_tensor has shape ', t.get_shape())
    corpus_reconstr = vocab.decode(corpus_ids)
    print('reconstructed corpus: ', corpus_reconstr)
    sess = tf.Session()
    t = Vocab.get_tensor(vocab)
    sess.run(tf.global_variables_initializer())
    print(sess.run(t))
    print('test pretrained embeddings:')
    for sym in vocab.symset_pt:
        id = vocab.sym2id[sym]
        emb = vocab.emb(sym)
        print('[{}] {} : {}'.format(id, sym, emb))
