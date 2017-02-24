from jtr.jack.core import *
from jtr.jack.data_structures import *

from jtr.preprocess.vocab import NeuralVocab

from abc import abstractmethod, ABCMeta
from typing import List, Tuple, Dict, Mapping

import tensorflow as tf

class SingleSupportFixedClassForward(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward_pass(self, shared_resources, nvocab,
                     Q, S, Q_lengths, S_lengths,
                     num_classes):
        '''Takes a single support and question and produces logits'''
        pass

#class MultipleSupportFixedClassForward(object):


class AbstractSingleSupportFixedClassModel(SimpleModelModule, SingleSupportFixedClassForward):
    def __init__(self, shared_resources):
        self.nvocab = None
        self.shared_resources = shared_resources

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.single_support,
                Ports.Input.question, Ports.Input.support_length,
                Ports.Input.question_length]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores,
                Ports.Prediction.candidate_idx]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores,
                Ports.Targets.candidate_idx]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    def create_output(self, shared_resources: SharedResources,
                      support : tf.Tensor,
                      question : tf.Tensor,
                      support_length : tf.Tensor,
                      question_length : tf.Tensor) -> Sequence[tf.Tensor]:

        if self.nvocab == None:
            self.nvocab = NeuralVocab(shared_resources.vocab,
                    input_size=shared_resources.config['repr_dim_input'])

        logits = self.forward_pass(shared_resources, self.nvocab,
                                   question, support, question_length,
                                   support_length,
                                   shared_resources.config['answer_size'])

        predictions = tf.arg_max(logits, 1, name='prediction')

        return [logits, predictions]


    def create_training_output(self, shared_resources: SharedResources,
                               logits : tf.Tensor,
                               labels : tf.Tensor) -> Sequence[tf.Tensor]:

        loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
            labels=labels), name='predictor_loss')
        return [loss]
