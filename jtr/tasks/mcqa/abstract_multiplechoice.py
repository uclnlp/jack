# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta
from typing import List

import tensorflow as tf

from jtr.core import *


class SingleSupportFixedClassForward(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward_pass(self, shared_resources,
                     Q_embedding_matrix, Q_ids, Q_lengths,
                     S_embedding_matrix, S_ids, S_lengths,
                     num_classes):
        '''Takes a single support and question and produces logits'''
        pass


class AbstractSingleSupportFixedClassModel(SimpleModelModule, SingleSupportFixedClassForward):
    def __init__(self, shared_resources, question_embedding_matrix=None, support_embedding_matrix=None):
        self.shared_resources = shared_resources
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config
        self.question_embedding_matrix = question_embedding_matrix
        self.support_embedding_matrix = support_embedding_matrix
        super(AbstractSingleSupportFixedClassModel, self).__init__(shared_resources)


    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.multiple_support,
                Ports.Input.question, Ports.Input.support_length,
                Ports.Input.question_length]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits,
                Ports.Prediction.candidate_index]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits,
                Ports.Target.target_index]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    def create_output(self, shared_resources: SharedResources,
                      support : tf.Tensor,
                      question : tf.Tensor,
                      support_length : tf.Tensor,
                      question_length : tf.Tensor) -> Sequence[tf.Tensor]:
        question_ids, support_ids = question, support
        if self.question_embedding_matrix is None:
            vocab_size = len(self.vocab)
            input_size = self.config['repr_dim_input']
            self.question_embedding_matrix = tf.get_variable(
                "emb_Q", [vocab_size, input_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True, dtype="float32")
            self.support_embedding_matrix = tf.get_variable(
                "emb_S", [vocab_size, input_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True, dtype="float32")

        logits = self.forward_pass(shared_resources,
                                   question_ids, question_length,
                                   support_ids, support_length,
                                   shared_resources.config['answer_size'])

        predictions = tf.arg_max(logits, 1, name='prediction')

        return [logits, predictions]


    def create_training_output(self, shared_resources: SharedResources,
                               logits : tf.Tensor, labels : tf.Tensor) -> Sequence[tf.Tensor]:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                              name='predictor_loss')
        return [loss]
