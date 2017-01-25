"""
Here we define the basic interfaces of jtr. jtr readers consist of 3 layers, one that transform
jtr data structures into tensors, one that processes predicts the outputs and losses
using a tensorflow model into other tensors, and one that converts these tensors back to jtr data structures.
"""

from abc import abstractmethod, ABCMeta, abstractproperty
from typing import Mapping, Iterable

import numpy as np
import tensorflow as tf

from jtr.data_structures import *


class TensorPort:
    """
    A TensorPort defines an input or output tensor for a ModelModule. This subsumes a
    shape of the tensor, and its data type.
    """

    def __init__(self, dtype, shape, name, doc_string=None, shape_string=None):
        self.shape_string = shape_string
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.__doc__ = doc_string

    def create_placeholder(self):
        """
        Convenience method that produces a placeholder of the type and shape defined by the port.
        Returns: a placeholder of same type, shape and name.

        """
        return tf.placeholder(self.dtype, self.shape, self.name)


class Ports:
    """
    This class groups input ports. Different modules can refer to these ports
    to define their input or output, respectively.
    """
    single_support = TensorPort(tf.int32, [None, None], "single_support",
                                "Represents instances with a single support document. "
                                "[batch_size, max_num_tokens]")
    multiple_support = TensorPort(tf.int32, [None, None, None], "multiple_support",
                                  "Represents instances with multiple support documents",
                                  "[batch_size, max_num_support, max_num_tokens")
    atomic_candidates = TensorPort(tf.int32, [None, None], "candidates",
                                   "Represents candidate choices using single symbolst",
                                   "[batch_size, num_candidates")
    question = TensorPort(tf.int32, [None, None], "question",
                          "Represents questions using symbol vectors",
                          "[batch_size, max_num_question_tokens]")
    scores = TensorPort(tf.float32, [None, None],
                        "Represents output scores for each candidate",
                        "[batch_size, num_candidates]")
    loss = TensorPort(tf.float32, [None],
                      "Represents loss on each instance in the batch",
                      "[batch_size]")


class InputModule(metaclass=ABCMeta):
    """
    An input module processes inputs and turns them into tensors to be processed by the model module.
    """

    @abstractproperty
    def output_ports(self) -> List[TensorPort]:
        pass

    @abstractmethod
    def __call__(self, inputs: List[Input]) -> Mapping[TensorPort, np.ndarray]:
        pass

    @abstractmethod
    def training_generator(self, training_set: List[(Input, Answer)]) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        pass


class ModelModule(metaclass=ABCMeta):
    """
    A model module encapsulates two tensorflow trees (possibly overlapping): a tree representing
    the answer prediction (to be processed by the outout module) and a tree representing the loss.
    It defines the expected input and output tensor shapes and types via its respective input
    and output pairs.
    """

    def __init__(self):
        self.placeholders = [d.create_placeholder() for d in self.input_ports()]
        self.predictions, self.loss = self.create(*self.placeholders)

    @abstractmethod
    def create(self, *tensors: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        This function needs to be implemented in order to define how the module produces
        output and loss tensors from input tensors.
        Args:
            *tensors: a list of input tensors. They are expected to match the order and type of ports in `input_ports`.

        Returns:
            output tensor, should match the shape and type of `output_port`
            loss tensor, should match the shape and type of `loss_port`

        """
        pass

    @abstractproperty
    def output_port(self) -> TensorPort:
        pass

    @abstractproperty
    def input_ports(self) -> List[TensorPort]:
        pass

    @abstractproperty
    def loss_port(self) -> TensorPort:
        pass

    @property
    def input_tensors(self) -> List[tf.Tensor]:
        return self.placeholders

    def convert_to_feed_dict(self, mapping: Mapping[TensorPort, np.ndarray]) -> Mapping[tf.Tensor, np.ndarray]:
        return {ph: mapping[self.input_ports[i]] for i, ph in self.input_tensors}

    @property
    def output_tensor(self) -> List[tf.Tensor]:
        return self.predictions

    @property
    def loss_tensor(self) -> tf.Tensor:
        return self.loss


class OutputModule(metaclass=ABCMeta):
    """
    An output module takes the output (numpy) tensors of the model module and turns them into
    jack data structures.
    """

    @abstractproperty
    def input_port(self) -> TensorPort:
        pass

    @abstractmethod
    def __call__(self, inputs: List[Input], prediction: np.ndarray) -> List[Answer]:
        pass


import jtr.train as jtr_train


class Reader:
    """
    A Reader reads inputs consisting of questions, supports and possibly candidates, and produces answers.
    It consists of three layers: input to tensor (input_module), tensor to tensor (model_module), and tensor to answer
    (output_model). These layers are called in-turn on a given input (list).
    """

    def __init__(self,
                 input_module: InputModule,
                 model_module: ModelModule,
                 output_module: OutputModule,
                 sess: tf.Session = None):
        self.sess = sess or tf.Session()
        self.output_module = output_module
        self.model_module = model_module
        self.input_module = input_module
        assert self.input_module.output_ports == self.model_module.input_ports, \
            "Input Module outputs must match model module inputs"

        assert self.model_module.output_port == self.output_module.input_port, \
            "Module model output must match output module inputs"

    def __call__(self, inputs: List[Input]) -> List[Answer]:
        """
        Reads all inputs (support and question), then returns an answer for each input.
        Args:
            inputs: a list of inputs.

        Returns: a list of answers.
        """
        batch = self.input_module(inputs)
        feed_dict = {input_tensor: batch[self.input_module.output_ports[i]]
                     for i, input_tensor in enumerate(self.model_module.input_tensors)}
        prediction = self.sess.run(self.model_module.output_tensor, feed_dict=feed_dict)
        answers = self.output_module(inputs, prediction)
        return answers

    def train(self, training_set: List[(Input, Answer)], **train_params):
        """
        This method trains the reader (and changes its state).
        Args:
            training_set: the training instances.
            **train_params: parameters to be sent to the training function `jtr.train.train`.

        Returns: None

        """
        train_port_mappings = self.input_module.training_generator(training_set)
        # note that this generator comprehension, not list comprehension
        train_feed_dicts = (self.model_module.convert_to_feed_dict(m) for m in train_port_mappings)
        args = {
            'loss': self.model_module.loss,
            'batches': train_feed_dicts,
            **train_params
        }
        jtr_train.train(**args)
