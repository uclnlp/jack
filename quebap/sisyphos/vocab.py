import tensorflow as tf
from quebap.sisyphos.tfutil import unit_length_transform, tfrun
import numpy as np


class Vocab(object):
    def __init__(self, unk="<UNK>", emb=None):
        if unk is not None:
            self.sym2id = {unk: 0}
            self.id2sym = {0:unk} #with pos and neg indices
            self.next_pos = 1
        else:
            self.sym2id = {}
            self.id2sym = {}
            self.next_pos = 0
        self.next_neg = -1
        self.sym2freqs = {}
        self.unk = unk
        self.emb = emb if emb is not None else lambda _:None #if emb is None: same behavior as for o-o-v words
        self.emb_length = None
        self.frozen = False

    def freeze(self):
        """map all ids to normalized ids"""
        if not self.frozen and self.next_neg < -1: #if any pretrained have been encountered
            sym2id = {sym: self._normalize(id) for sym,id in self.sym2id.items()}
            id2sym = {self._normalize(id): sym for id,sym in self.id2sym.items()}
            self.sym2id = sym2id
            self.id2sym = id2sym
        self.frozen = True

    def unfreeze(self):
        """map all normalized ids to original (pos/neg) ids"""
        if self.frozen and self.next_neg < -1:
            sym2id = {sym: self._denormalize(id) for sym, id in self.sym2id.items()}
            id2sym = {self._denormalize(id): sym for id, sym in self.id2sym.items()}
            self.sym2id = sym2id
            self.id2sym = id2sym
        self.frozen = False

    def get_id(self, sym):
        """Note: different behavior for self.frozen == True vs. False!
           if not self.frozen: negative id's for symbols with pre-trained embedding; ad's added if need be
           if self.frozen: symbols with pre-trained embedding get positive id
        """
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
                    self.emb_length = len(vec)
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

    def _normalize(self,id):
        """map original (pos/neg) ids to normalized (non-neg) ids: first new symbols, then those in emb"""
        #e.g. -1 should be mapped to self.next_pos + 0
        #e.g. -3 should be mapped to self.next_pos + 2
        return id if id >=0 else self.next_pos - id - 1

    def _denormalize(self,id):
        #self.next_pos + i is mapped back to  -1-i
        return id if id < self.next_pos else -1-(id-self.next_pos)

    def get_ids_pretrained(self):
        if self.frozen:
            return list(range(self.next_pos,self.next_pos+self.count_pretrained()))
        else:
            return list(range(-1,self.next_neg,-1))

    def get_ids_oov(self):
        return list(range(self.next_pos))

    def count_pretrained(self):
        return -self.next_neg - 1

    def count_oov(self):
        return self.next_pos




# #original class
# class Vocab(object):
#     def __init__(self, unk="<UNK>"):
#         if unk is not None:
#             self.sym2id = {unk: 0}
#             self.id2sym = [unk]
#         else:
#             self.sym2id = {}
#             self.id2sym = []
#         self.sym2freqs = {}
#         self.unk = unk
#         self.frozen = False
#
#     def freeze(self):
#         self.frozen = True
#
#     def unfreeze(self):
#         self.frozen = False
#
#     def get_id(self, sym):
#         if not self.frozen:
#             if sym not in self.sym2id:
#                 self.sym2id[sym] = len(self.id2sym)
#                 self.id2sym.append(sym)
#                 self.sym2freqs[sym] = 1
#             else:
#                 self.sym2freqs[sym] += 1
#         if sym in self.sym2id:
#             return self.sym2id[sym]
#         else:
#             return self.sym2id[self.unk]
#
#     def get_sym(self, id):
#         return self.id2sym[id]
#
#     def __call__(self, *args, **kwargs):
#         symbols = args
#         if len(args) == 1:
#             if isinstance(args[0], list):
#                 symbols = args[0]
#             else:
#                 return self.get_id(args[0])
#
#         return [self.get_id(sym) for sym in symbols]
#
#     def __len__(self):
#         return len(self.id2sym)
#
#     def __contains__(self, sym):
#         return sym in self.sym2id


class NeuralVocab(Vocab):
    """
    Wrapper around Vocab to go from indices to tensors.
    TODO: documentation

    :param embedding_matrix: tensor with shape (len_vocab,input_size). If provided,
    the arguments input_size, use_trained, and train_pretrained are ignored,
    but unit_normalize is still performed if unit_normalize==True.
    :param input_size: integer; embedding length in case embedding_matrix not provided, otherwise ignored.
    :param use_pretrained: boolean; True (default): use pre-trained if available (through Vocab.emb).
    False: do not explicitly assign embeddings from vocab.emb (may or may not be pre-set in embedding_matrix).
    Ignored if embedding_matrix is given.
    :param train_pretrained: boolean; False (default): fix pretrained embeddings; True: continue training.
    Ignored if embedding_matrix is given.
    :param unit_normalize: normalize embedding vectors (including pretrained ones)
    """
    def __init__(self, base_vocab, embedding_matrix=None,
                 input_size=None, use_pretrained=True, train_pretrained=False, unit_normalize=False):
        print(base_vocab)
        print(base_vocab.unk)
        super(NeuralVocab, self).__init__(unk=base_vocab.unk, emb=base_vocab.emb)

        assert (embedding_matrix, input_size) is not (None, None), "if no embedding_matrix is provided, define input_size"

        base_vocab.freeze() #freeze if not frozen (to ensure fixed non-negative indices)

        self.sym2id = base_vocab.sym2id
        self.id2sym = base_vocab.id2sym
        self.sym2freqs = base_vocab.sym2freqs
        self.unit_normalize = unit_normalize

        if embedding_matrix is None:
            #construct part oov
            n_oov = base_vocab.count_oov()
            n_pre = base_vocab.count_pretrained()
            E_oov = tf.get_variable("embeddings_oov", [n_oov, input_size],
                                     initializer=tf.random_normal_initializer(0, 0.1),
                                     trainable=True, dtype="float32")

            #construct part pretrained
            if use_pretrained and base_vocab.emb_length is not None:
                #load embeddings into numpy tensor with shape (count_pretrained, min(input_size,emb_length))
                np_E_pre = np.zeros([n_pre, min(input_size, base_vocab.emb_length)]).astype("float32")
                for id in base_vocab.get_ids_pretrained():
                    sym = base_vocab.id2sym[id]
                    i = id - n_oov  #shifted to start from 0
                    np_E_pre[i,:] = base_vocab.emb(sym)[:min(input_size,base_vocab.emb_length)]
                E_pre = tf.get_variable("embeddings_pretrained", initializer=tf.identity(np_E_pre),
                                        trainable=train_pretrained, dtype="float32")
                print('test base_vocab.emb_length ',base_vocab.emb_length)
                print('test input_size ',input_size)
                print('test E_pre: ',E_pre.get_shape())
                if input_size > base_vocab.emb_length:
                    E_pre_ext = tf.get_variable("embeddings_extra", [n_pre, input_size-base_vocab.emb_length],
                        initializer=tf.random_normal_initializer(0.0, 0.1), dtype="float32", trainable=True)
                    print('test E_pre_ext: ', E_pre_ext.get_shape())
                    E_pre = tf.concat(1, [E_pre, E_pre_ext], name="embeddings_pretrained_extended")

            else:
                E_pre = tf.get_variable("embeddings_not_pretrained", [n_pre, input_size],
                                        initializer=tf.random_normal_initializer(-0.05, 0.05),
                                        trainable=True, dtype="float32")

            self.input_size = input_size   #must be provided is embedding_matrix is None
            self.embedding_matrix = tf.concat(0, [E_oov, E_pre], name="embeddings")

        else:
            self.input_size = embedding_matrix.get_shape()[1] #ignore input argument input_size
            self.embedding_matrix = embedding_matrix

        if self.unit_normalize:
            self.embedding_matrix = unit_length_transform(self.embedding_matrix, dim=1)

        #pre-assign embedding vectors to all ids
        self.id2vec = [self.embed_symbol(id) for id in range(len(self))] #always OK if frozen



    def embed_symbol(self, id):
        id_embedded = tf.nn.embedding_lookup(self.embedding_matrix, id)
        return id_embedded


    def __call__(self, *args, **kwargs):
        if len(args) == 1:  #tuple with length 1: then either list with ids, tensor with ids, or single id
            if isinstance(args[0], list):
                ids = args[0]
            elif tf.contrib.framework.is_tensor(args[0]):
                #return embedded tensor
                return tf.nn.embedding_lookup(self.embedding_matrix, args[0])
            else:
                return self.id2vec[args[0]]
        else: #tuple with ids
            ids = args
        return [self.id2vec[id] for id in ids]




if __name__ == '__main__':

    print(40*'-'+'\n(1) TEST Vocab without pretrained embeddings\n'+40*'-')
    vocab = Vocab()
    print(vocab("blah"))
    print(vocab("bluh"))
    print(vocab("bleh"))
    print(vocab("bluh"))
    print(vocab("hello"))
    print(vocab("world"))
    print('sym2id before freezing ',vocab.sym2id)
    vocab.freeze()
    print('sym2id after freezing ',vocab.sym2id)


    def emb(w):
        v = {'blah':[1.7,0,.3],'bluh':[0,1.5,0.5],'bleh':[0,0,2]}
        if w in v:
            return v[w]
        else:
            return None
    print(40*'-'+'\n(2) TEST Vocab with pretrained embeddings\n'+40*'-')

    vocab = Vocab(emb=emb)
    print(vocab("blah"))
    print(vocab("bluh"))
    print(vocab("bleh"))
    print(vocab("bluh"))
    print(vocab("hello"))
    print(vocab("world"))
    print('sym2id before freezing ',vocab.sym2id)
    print('test ids_pretrained %s and ids_new %s before freezing'%(str(vocab.get_ids_pretrained()),str(vocab.get_ids_oov())))
    vocab.freeze()
    print('sym2id after freezing ',vocab.sym2id)
    print('test ids_pretrained %s and ids_new %s after freezing'%(str(vocab.get_ids_pretrained()),str(vocab.get_ids_oov())))
    print("(frozen) vocab(['bluh','world','wake','up']);  last two are new words")
    print(vocab(['bluh','world','wake','up']))
    vocab.unfreeze()
    print("(unfrozen) vocab(['bluh','world','wake','up']);  last two are new words")
    print(vocab(['bluh','world','wake','up']))
    print('(unfrozen) test ids_pretrained %s and ids_new %s'%(str(vocab.get_ids_pretrained()),str(vocab.get_ids_oov())))

    print(vocab.get_sym(1))
    print(vocab.get_sym(-1))

    print(40*'-'+'\n(3) TEST NeuralVocab without pretrained embeddings\n'+40*'-')
    with tf.variable_scope('neural_test1'):
        nvocab = NeuralVocab(vocab, None, 3, unit_normalize=True)
        vec = tfrun(nvocab(vocab("world")))
        print(vec)

    print(40 * '-' + '\n(4) TEST NeuralVocab with pretrained embeddings\n' + 40 * '-')
    with tf.variable_scope('neural_test2'):
        print('pretrained: ')
        for w in ['blah','bluh','bleh']:
            print('\t%s : %s'%(w,str(emb(w))))
        print('construct NeuralVocab with input_size 4, with length-3 pre-trained embeddings')
        nvocab = NeuralVocab(vocab, None, 4, unit_normalize=True, use_pretrained=True, train_pretrained=False)
        print('embedding matrix:')
        mat = tfrun(nvocab.embedding_matrix)
        print(mat)
