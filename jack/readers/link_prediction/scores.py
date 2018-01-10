# -*- coding: utf-8 -*-

import abc
import sys

import tensorflow as tf

from jack.readers.link_prediction.similarities import negative_l1_distance


class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self, subject_embeddings=None, predicate_embeddings=None, object_embeddings=None, *args, **kwargs):
        """
        Abstract class inherited by all models.

        Args:
            subject_embeddings: (batch_size, entity_embedding_size) Tensor.
            predicate_embeddings: (batch_size, predicate_embedding_size) Tensor.
            object_embeddings: (batch_size, entity_embedding_size) Tensor.
        """
        self.subject_embeddings = subject_embeddings
        self.predicate_embeddings = predicate_embeddings
        self.object_embeddings = object_embeddings

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    @property
    def parameters(self):
        return []


class TranslatingModel(BaseModel):
    def __init__(self, similarity_function=negative_l1_distance, *args, **kwargs):
        """
        Implementation of the Translating Embeddings model [1].
        [1] Bordes, A. et al. - Translating Embeddings for Modeling Multi-relational Data - NIPS 2013

        Args:
            similarity_function: Similarity function.
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
    def __init__(self, *args, **kwargs):
        """
        Implementation of the Bilinear-Diagonal model [1]
        [1] Yang, B. et al. - Embedding Entities and Relations for Learning and Inference in Knowledge Bases - ICLR 2015

        Args:
            similarity_function: Similarity function.
        """
        super().__init__(*args, **kwargs)

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        scaled_subject_embedding = self.subject_embeddings * self.predicate_embeddings
        return tf.reduce_sum(scaled_subject_embedding * self.object_embeddings, axis=1)


class BilinearModel(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        Implementation of the Bilinear model [1]
        [1] Nickel, M. et al. - A Three-Way Model for Collective Learning on Multi-Relational Data - ICML 2011

        ArgS:
            similarity_function: Similarity function.
        """
        super().__init__(*args, **kwargs)

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        es, emb_size = tf.expand_dims(self.subject_embeddings, 1), tf.shape(self.subject_embeddings)[1]
        W = tf.reshape(self.predicate_embeddings, (-1, emb_size, emb_size))
        sW = tf.matmul(es, W)[:, 0, :]
        return tf.reduce_sum(sW * self.object_embeddings, axis=1)


class ComplexModel(BaseModel):
    def __init__(self, *args, **kwargs):
        """
        Implementation of the ComplEx model [1]
        [1] Trouillon, T. et al. - Complex Embeddings for Simple Link Prediction - ICML 2016

        Args:
            embedding size: Embedding size.
        """
        super().__init__(*args, **kwargs)

    def __call__(self):
        """
        :return: (batch_size) Tensor containing the scores associated by the models to the walks.
        """
        es_re, es_im = tf.split(value=self.subject_embeddings, num_or_size_splits=2, axis=1)
        eo_re, eo_im = tf.split(value=self.object_embeddings, num_or_size_splits=2, axis=1)
        ew_re, ew_im = tf.split(value=self.predicate_embeddings, num_or_size_splits=2, axis=1)

        def dot3(arg1, rel, arg2):
            return tf.reduce_sum(arg1 * rel * arg2, axis=1)

        score = dot3(es_re, ew_re, eo_re) + dot3(es_re, ew_im, eo_im) + dot3(es_im, ew_re, eo_im) - dot3(es_im, ew_im, eo_re)
        return score

# Aliases
TransE = TranslatingEmbeddings = TranslatingModel
DistMult = BilinearDiagonal = BilinearDiagonalModel
RESCAL = Bilinear = BilinearModel
ComplEx = ComplexE = ComplexModel


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown model: {}'.format(function_name))
    return getattr(this_module, function_name)
