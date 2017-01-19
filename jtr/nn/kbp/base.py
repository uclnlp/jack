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
        sW = tf.squeeze(tf.batch_matmul(es, self.predicate_embeddings), axis=1)

        return self.similarity_function(sW, self.object_embeddings)


# Aliases
TransE = TranslatingEmbeddings = TranslatingModel
DistMult = BilinearDiagonal = BilinearDiagonalModel
RESCAL = Bilinear = BilinearModel
