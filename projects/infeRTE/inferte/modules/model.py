# -*- coding: utf-8 -*-

from typing import Dict, Any
from abc import abstractmethod
from typing import List, Mapping, Sequence

import tensorflow as tf

from jtr.jack.core import Ports, TensorPort, ModelModule
from jtr.jack.tf_fun import rnn, simple

from jtr.jack.tasks.mcqa.abstract_multiplechoice import SingleSupportFixedClassForward


class SimpleModelModule(ModelModule):
    """
    This class simplifies the implementation of ModelModules by requiring to implement a small set of methods that
    produce the TF graphs to create predictions and the training outputs, and define the ports.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def create_output(self, config: Dict[str, Any],
                      *input_tensors: tf.Tensor) -> Sequence[tf.Tensor]:
        """
        This function needs to be implemented in order to define how the module produces
        output from input tensors corresponding to `input_ports`.
        Args:
            *input_tensors: a list of input tensors.

        Returns:
            mapping from defined output ports to their tensors.
        """
        pass

    @abstractmethod
    def create_training_output(self, config: Dict[str, Any],
                               *training_input_tensors: tf.Tensor) -> Sequence[tf.Tensor]:
        """
        This function needs to be implemented in order to define how the module produces tensors only used
        during training given tensors corresponding to the ones defined by `training_input_ports`, which might include
        tensors corresponding to ports defined by `output_ports`. This sub-graph should only be created during training.
        Args:
            *training_input_tensors: a list of input tensors.

        Returns:
            mapping from defined training output ports to their tensors.
        """
        pass

    def setup(self, is_training=True):
        old_train_variables = tf.trainable_variables()
        old_variables = tf.global_variables()
        self._tensors = {d: d.create_placeholder() for d in self.input_ports}
        self._placeholders = dict(self._tensors)
        output_tensors = self.create_output(self.config, *[self._tensors[port] for port in self.input_ports])
        self._tensors.update(zip(self.output_ports, output_tensors))
        if is_training:
            self._placeholders.update((p, p.create_placeholder()) for p in self.training_input_ports
                                      if p not in self._placeholders and p not in self._tensors)
            self._tensors.update(self._placeholders)
            input_target_tensors = {p: self._tensors.get(p, None) for p in self.training_input_ports}
            training_output_tensors = self.create_training_output(self.config, *[input_target_tensors[port]
                                                                                           for port in
                                                                                           self.training_input_ports])
            self._tensors.update(zip(self.training_output_ports, training_output_tensors))
        self._training_variables = [v for v in tf.trainable_variables() if v not in old_train_variables]
        self._saver = tf.train.Saver(self._training_variables, max_to_keep=1)
        self._variables = [v for v in tf.global_variables() if v not in old_variables]

    @property
    def placeholders(self) -> Mapping[TensorPort, tf.Tensor]:
        return self._placeholders

    @property
    def tensors(self) -> Mapping[TensorPort, tf.Tensor]:
        """
        Returns: Mapping from ports to tensors
        """
        return self._tensors if hasattr(self, "_tensors") else None

    def store(self, sess, path):
        self._saver.save(sess, path)

    def load(self, sess, path):
        self._saver.restore(sess, path)

    @property
    def train_variables(self) -> Sequence[tf.Tensor]:
        """ Returns: A list of training variables """
        return self._training_variables

    @property
    def variables(self) -> Sequence[tf.Tensor]:
        """ Returns: A list of variables """
        return self._variables


class AbstractSingleSupportFixedClassModel(SimpleModelModule, SingleSupportFixedClassForward):
    def __init__(self, config,
                 question_embedding_matrix=None,
                 support_embedding_matrix=None):
        self.config = config
        self.question_embedding_matrix = question_embedding_matrix
        self.support_embedding_matrix = support_embedding_matrix

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

    def create_output(self, config: Dict[str, Any],
                      support : tf.Tensor,
                      question : tf.Tensor,
                      support_length : tf.Tensor,
                      question_length : tf.Tensor) -> Sequence[tf.Tensor]:
        question_ids, support_ids = question, support
        if self.question_embedding_matrix is None:
            vocab_size = self.config['vocab_size']
            input_size = self.config['repr_dim_input']
            self.question_embedding_matrix = tf.get_variable(
                "emb_Q", [vocab_size, input_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True, dtype="float32")
            self.support_embedding_matrix = tf.get_variable(
                "emb_S", [vocab_size, input_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True, dtype="float32")

        logits = self.forward_pass(config,
                                   question_ids, question_length,
                                   support_ids, support_length,
                                   self.config['answer_size'])

        predictions = tf.arg_max(logits, 1, name='prediction')

        return [logits, predictions]


    def create_training_output(self, config: Dict[str, Any],
                               logits : tf.Tensor, labels : tf.Tensor) -> Sequence[tf.Tensor]:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels), name='predictor_loss')
        return [loss]


class PairOfBiLSTMOverSupportAndQuestionModel(AbstractSingleSupportFixedClassModel):
    def forward_pass(self, config,
                     Q_ids, Q_lengths,
                     S_ids,  S_lengths,
                     num_classes):
        # final states_fw_bw dimensions:
        # [[[batch, output dim], [batch, output_dim]]
        S_ids = tf.squeeze(S_ids, 1)
        S_lengths = tf.squeeze(S_lengths, 1)

        Q_seq = tf.nn.embedding_lookup(self.question_embedding_matrix, Q_ids)
        S_seq = tf.nn.embedding_lookup(self.support_embedding_matrix, S_ids)

        all_states_fw_bw, final_states_fw_bw = rnn.pair_of_bidirectional_LSTMs(
            Q_seq, Q_lengths, S_seq, S_lengths, config['repr_dim'],
            drop_keep_prob=1.0 - config['dropout'],
            conditional_encoding=True)
        # ->  [batch, 2*output_dim]
        final_states = tf.concat([final_states_fw_bw[0][1], final_states_fw_bw[1][1]],axis=1)
        # [batch, 2*output_dim] -> [batch, num_classes]
        outputs = simple.fully_connected_projection(final_states, num_classes)
        return outputs
