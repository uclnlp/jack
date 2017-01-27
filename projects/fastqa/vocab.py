from jtr.preprocess.vocab import Vocab
import tensorflow as tf
import numpy as np


class EmbeddingNeuralVocab(Vocab):
    """
    This is a very simple implementation of a neural vocab based on predefined embeddings
    """

    def __init__(self, embeddings, trainable=False):
        """
        Args:
            embeddings: loaded embeddings instance

        Returns:

        """
        super(EmbeddingNeuralVocab, self).__init__(emb=embeddings)

        self.sym2id = embeddings.vocabulary.word2idx
        self.id2sym = embeddings.vocabulary.idx2word
        self.emb_length = embeddings.lookup.shape[1]

        if Vocab.DEFAULT_UNK not in self.sym2id:
            self.sym2id[Vocab.DEFAULT_UNK] = len(self.sym2id)
            self.id2sym.append(Vocab.DEFAULT_UNK)
            embeddings.lookup = np.concatenate([embeddings.lookup, np.zeros([1, self.emb_length])])

        self.frozen = True
        self.embedding_matrix = tf.get_variable("embedding_matrix", initializer=embeddings.lookup, trainable=False)

    def embed_symbol(self, ids):
        """returns embedded id's

        Args:
            `ids`: integer, ndarray with np.int32 integers, or tensor with tf.int32 integers.
            These integers correspond to (normalized) id's for symbols in `self.base_vocab`.

        Returns:
            tensor with id's embedded by numerical vectors (in last dimension)
        """
        return tf.nn.embedding_lookup(self.embedding_matrix, ids)

    def get_embedding_matrix(self):
        return self.embedding_matrix
