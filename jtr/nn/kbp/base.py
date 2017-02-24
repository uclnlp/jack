# -*- coding: utf-8 -*-

import abc
import tensorflow as tf

from .similarities import negative_l1_distance, dot_product


class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self, subject_embeddings=None, predicate_embeddings=None, object_embeddings=None,
                 *args, **kwargs):
        """
        Abstract class inherited by all models.

        :param subject_embeddings: (batch_size, entity_embedding_size) Tensor.
        :param predicate_embeddings: (batch_size, predicate_embedding_size) Tensor.
        :param subject_embeddings: (batch_size, entity_embedding_size) Tensor.
        """
        self.subject_embeddings = subject_embeddings
        self.predicate_embeddings = predicate_embeddings
        self.object_embeddings = object_embeddings

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    def get_params(self):
        return []


class TranslatingModel(BaseModel):
    def __init__(self, similarity_function=negative_l1_distance, *args, **kwargs):
        """
        Implementation of a compositional extension of the Translating Embeddings model [1].
        [1] Bordes, A. et al. - Translating Embeddings for Modeling Multi-relational Data - NIPS 2013

        :param similarity_function: Similarity function.
        """
        super().__init__(*args, **kwargs)
        self.similarity_function = similarity_function

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        translated_subject_embedding = self.subject_embeddings + self.predicate_embeddings
        return self.similarity_function(translated_subject_embedding, self.object_embeddings)


class BilinearDiagonalModel(BaseModel):
    def __init__(self, similarity_function=dot_product, *args, **kwargs):
        """
        Implementation of a compositional extension of the Bilinear-Diagonal model [1]
        [1] Yang, B. et al. - Embedding Entities and Relations for Learning and Inference in Knowledge Bases - ICLR 2015

        :param similarity_function: Similarity function.
        """
        super().__init__(*args, **kwargs)
        self.similarity_function = similarity_function

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        scaled_subject_embedding = self.subject_embeddings * self.predicate_embeddings
        return self.similarity_function(scaled_subject_embedding, self.object_embeddings)


class BilinearModel(BaseModel):
    def __init__(self, similarity_function=dot_product, *args, **kwargs):
        """
        Implementation of a compositional extension of the Bilinear model [1]
        [1] Nickel, M. et al. - A Three-Way Model for Collective Learning on Multi-Relational Data - ICML 2011

        :param similarity_function: Similarity function.
        """
        super().__init__(*args, **kwargs)
        self.similarity_function = similarity_function

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        es = tf.expand_dims(self.subject_embeddings, 1)
        emb_size = tf.shape(self.subject_embeddings)[1]
        W = tf.reshape(self.predicate_embeddings, (-1, emb_size, emb_size))
        sW = tf.matmul(es, W)[:, 0, :]

        return self.similarity_function(sW, self.object_embeddings)


class ComplexModel(BaseModel):
    def __init__(self, embedding_size=None, *args, **kwargs):
        """
        Implementation of a compositional extension of the ComplEx model [1]
        [1] Trouillon, T. et al. - Complex Embeddings for Simple Link Prediction - ICML 2016

        :param embedding size: Embedding size.
        """
        super().__init__(*args, **kwargs)
        self.embedding_size = embedding_size

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        es_re, es_im = self.subject_embeddings[:, :self.embedding_size], self.subject_embeddings[:, self.embedding_size:]
        eo_re, eo_im = self.object_embeddings[:, :self.embedding_size], self.object_embeddings[:, self.embedding_size:]
        ew_re, ew_im = self.predicate_embeddings[:, :self.embedding_size], self.predicate_embeddings[:, self.embedding_size:]

        def dot3(arg1, rel, arg2):
            return tf.reduce_sum(arg1 * rel * arg2, axis=1)

        score = dot3(es_re, ew_re, eo_re) + dot3(es_re, ew_im, eo_im) + dot3(es_im, ew_re, eo_im) - dot3(es_im, ew_im, eo_re)
        return score


class ERMLP(BaseModel):
    def __init__(self, entity_embedding_size=None, predicate_embedding_size=None,
                 hidden_size=None, f=tf.tanh, *args, **kwargs):
        """
        Implementation of the ER-MLP model described in [1, 2]

        [1] Dong, X. L. et al. - Knowledge Vault: A Web-Scale Approach to Probabilistic Knowledge Fusion - KDD 2014
        [2] Nickel, M. et al. - A Review of Relational Machine Learning for Knowledge Graphs - IEEE 2016

        :param entity_embedding_size: Entity embedding size.
        :param predicate_embedding_size: Predicate embedding size.
        :param hidden_size: Hidden size of the MLP.
        :param f: non-linearity of the MLP.
        """
        super().__init__(*args, **kwargs)
        self.f = f

        self.entity_embedding_size = entity_embedding_size
        self.predicate_embedding_size = predicate_embedding_size
        input_size = self.entity_embedding_size + self.entity_embedding_size + self.predicate_embedding_size

        self.C = tf.get_variable('C', shape=[input_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        self.w = tf.get_variable('w', shape=[hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        e_ijk = tf.concat([self.subject_embeddings, self.object_embeddings,
            self.predicate_embeddings], 1)
        h_ijk = tf.matmul(e_ijk, self.C)
        f_ijk = tf.squeeze(tf.matmul(self.f(h_ijk), self.w), axis=1)

        return f_ijk

    def get_params(self):
        params = super().get_params() + [self.C, self.w]
        return params


# Aliases
TransE = TranslatingEmbeddings = TranslatingModel
DistMult = BilinearDiagonal = BilinearDiagonalModel
RESCAL = Bilinear = BilinearModel
ComplEx = ComplexE = ComplexModel
ER_MLP = ERMLP
