import tensorflow as tf
from sisyphos.tfutil import unit_length_transform, tfrun


class VocabEmb(object):
    def __init__(self, unk="<UNK>", emb=None):
        if unk is not None:
            self.sym2id = {unk: 0}
            self.id2sym = {0:unk} #with pos and neg indices
            self.next_pos = 1
            self.next_neg = -1
        else:
            self.sym2id = {}
            self.id2sym = {}
            self.next_pos = 0
            self.next_neg = -1
        self.sym2freqs = {}
        self.unk = unk
        self.emb = emb if emb is not None else lambda _:None #if emb is None: same behavior as for o-o-v words
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def get_id(self, sym):
        if not self.frozen:
            vec = self.emb(sym)
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
            return self.sym2id[self.unk]

    def get_sym(self, id):
        return None if not id in self.id2sym else self.id2sym[id]

    def __call__(self, *args, **kwargs):
        symbols = args
        if len(args) == 1:
            if isinstance(args[0], list):
                symbols = args[0]
            else:
                return self.get_id(args[0])

        return [self.get_id(sym) for sym in symbols]

    def __len__(self):
        return len(self.id2sym)

    def __contains__(self, sym):
        return sym in self.sym2id

    def normalize(self,id):
        #e.g. -3 should be mapped to self.next_pos + 2
        return id if id >=0 else self.next_pos - id - 1

    def count_pretrained(self):
        return -self.next_neg - 1
    def count_oov(self):
        return self.next_pos




#original class
class Vocab(object):
    def __init__(self, unk="<UNK>"):
        if unk is not None:
            self.sym2id = {unk: 0}
            self.id2sym = [unk]
        else:
            self.sym2id = {}
            self.id2sym = []
        self.sym2freqs = {}
        self.unk = unk
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def get_id(self, sym):
        if not self.frozen:
            if sym not in self.sym2id:
                self.sym2id[sym] = len(self.id2sym)
                self.id2sym.append(sym)
                self.sym2freqs[sym] = 1
            else:
                self.sym2freqs[sym] += 1
        if sym in self.sym2id:
            return self.sym2id[sym]
        else:
            return self.sym2id[self.unk]

    def get_sym(self, id):
        return self.id2sym[id]

    def __call__(self, *args, **kwargs):
        symbols = args
        if len(args) == 1:
            if isinstance(args[0], list):
                symbols = args[0]
            else:
                return self.get_id(args[0])

        return [self.get_id(sym) for sym in symbols]

    def __len__(self):
        return len(self.id2sym)

    def __contains__(self, sym):
        return sym in self.sym2id


class NeuralVocab(Vocab):
    """
    Wrapper around Vocab to go from indices to tensors.
    """
    def __init__(self, base_vocab, embedding_matrix=None,
                 input_size=None, unit_normalize=False):
        super(NeuralVocab, self).__init__()
        self.sym2id = base_vocab.sym2id
        self.id2sym = base_vocab.id2sym
        self.sym2freqs = base_vocab.sym2freqs
        self.unk = base_vocab.unk

        self.unit_normalize = unit_normalize

        if embedding_matrix is None:
            self.embedding_matrix = \
                tf.get_variable("embeddings", [len(self), input_size],
                        initializer=tf.random_normal_initializer(0.0, 0.1))
        else:
            self.embedding_matrix = embedding_matrix

        self.id2vec = [self.embed_symbol(id) for id in range(len(self))]

    def embed_symbol(self, id):
        id_embedded = tf.nn.embedding_lookup(self.embedding_matrix, id)
        if self.unit_normalize:
            id_embedded = unit_length_transform(id_embedded, 0)
        return id_embedded

    def __call__(self, *args, **kwargs):
        ids = args
        if len(args) == 1:
            if isinstance(args[0], list):
                ids = args[0]
            else:
                return self.id2vec[args[0]]

        return [self.id2vec[id] for id in ids]


if __name__ == '__main__':
    print('TEST Vocab')
    vocab = Vocab()
    print(vocab("blah"))
    print(vocab("bluh"))
    print(vocab("bleh"))
    print(vocab("bluh"))
    vocab.freeze()
    print(vocab("bluh"))
    print(vocab("what?"))
    vocab.unfreeze()
    print(vocab("what?"))
    print(vocab("I'm", "afraid", "I", "can't", "do", "that", "dave", "!"))
    print(vocab(["all", "the", "work"]))
    print(vocab.get_sym(10))
    print(len(vocab))

    print('TEST VocabEmb')
    def emb(w):
        v = {'blah':[1,0,0],'bluh':[0,1,0],'bleh':[0,0,1]}
        if w in v:
            return v[w]
        else:
            return None

    vocab = VocabEmb(emb=emb)
    print(vocab("blah"))
    print(vocab("bluh"))
    print(vocab("bleh"))
    print(vocab("bluh"))
    print(vocab("hello"))
    print(vocab("world"))
    vocab.freeze()
    print(vocab(['bluh','world','wake','up']))
    print(vocab.get_sym(1))
    print(vocab.get_sym(-1))
    print('test normalizing %s'%str(vocab(['bluh','world','wake','up'])))
    print(list(map(vocab.normalize,vocab(['bluh','world','wake','up']))))
    print(vocab.sym2id)

    nvocab = NeuralVocab(vocab, None, 3, unit_normalize=True)
    vec = tfrun(nvocab(vocab("world")))
    print(vec)
