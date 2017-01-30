"""
Here we define the basic interfaces of jtr. jtr readers consist of 3 layers, one that transform
jtr data structures into tensors, one that processes predicts the outputs and losses
using a tensorflow model into other tensors, and one that converts these tensors back to jtr data structures.
"""

from abc import abstractmethod, ABCMeta, abstractproperty
from typing import Mapping, Iterable, Tuple

import numpy as np
import tensorflow as tf
import jtr.train as jtr_train

from jtr.data_structures import *
from jtr.util.hooks import LossHook

import logging
import sys


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

    def __gt__(self, port):
        return self.name > port.name


class Ports:
    """
    This class groups input ports. Different modules can refer to these ports
    to define their input or output, respectively.
    """
    single_support = TensorPort(tf.int32, [None, None], "single_support",
                                "Represents instances with a single support document. ",
                                "[batch_size, max_num_tokens]")
    multiple_support = TensorPort(tf.int32, [None, None, None], "multiple_support",
                                  "Represents instances with multiple support documents",
                                  "[batch_size, max_num_support, max_num_tokens]")
    atomic_candidates = TensorPort(tf.int32, [None, None], "candidates",
                                   "Represents candidate choices using single symbols",
                                   "[batch_size, num_candidates]")
    question = TensorPort(tf.int32, [None, None], "question",
                          "Represents questions using symbol vectors",
                          "[batch_size, max_num_question_tokens]")
    scores = TensorPort(tf.float32, [None, None], "scores",
                        "Represents output scores for each candidate",
                        "[batch_size, num_candidates]")
    loss = TensorPort(tf.float32, [None], "loss",
                      "Represents loss on each instance in the batch",
                      "[batch_size]")

    candidate_targets = TensorPort(tf.float32, [None, None], "candidate_targets",
                                   "Represents target (0/1) values for each candidate",
                                   "[batch_size, num_candidates]")


class SharedResources(metaclass=ABCMeta):
    """
    A class to store explicit shared resources between layers. It is recommended to minimise information stored here.
    """

    @abstractmethod
    def store(self):
        pass


class SharedVocab(SharedResources):
    """
    A class to provide and store a vocab shared across some of the reader modules.
    """

    def __init__(self, vocab, config=None):
        self.config = config
        self.vocab = vocab

    def store(self):
        # todo: store vocab to file location specified in vocab
        pass


class Module(metaclass=ABCMeta):
    """
    Class to specify shared signature between modules.
    """

    @abstractmethod
    def store(self):
        """
        Store the state of this module.
        """
        pass


class InputModule(Module):
    """
    An input module processes inputs and turns them into tensors to be processed by the model module.
    """

    @abstractproperty
    def output_ports(self) -> List[TensorPort]:
        """
        Defines what types of tensors the output module produces in each batch.
        Returns: a list of tensor ports that correspond to the tensor ports in the mapping
        produced by `__call__`.
        """
        pass

    @abstractproperty
    def target_port(self) -> TensorPort:
        """
        Defines what type of tensor is used to represent the target solution during training.
        """
        pass

    @abstractmethod
    def __call__(self, inputs: List[Input]) -> Mapping[TensorPort, np.ndarray]:
        """
        Converts a list of inputs into a single batch of tensors, consisting with the `output_ports` of this
        module.
        Args:
            inputs: a list of instances (question, support, optional candidates)

        Returns:
            A mapping from ports to tensors.

        """
        pass

    @abstractmethod
    def training_generator(self, training_set: List[Tuple[Input, Answer]]) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        """
        Given a training set of input-answer pairs, this method produces an iterable/generator
        that when iterated over returns a sequence of batches. These batches map ports to tensors
        just as `__call__` does, using the `output_ports` of this object and the `target_port`.
        Args:
            training_set: a set of pairs of input and answer.

        Returns: An iterable/generator that, on each pass through the data, produces a sequence of batches.
        """
        pass

    @abstractmethod
    def setup(self, data: List[Tuple[Input, Answer]]) -> SharedVocab:
        """
        Args:
            data: a set of pairs of input and answer.

        Returns: vocab
        """
        pass


class ModelModule(Module):
    """
    A model module encapsulates two tensorflow trees (possibly overlapping): a tree representing
    the answer prediction (to be processed by the outout module) and a tree representing the loss.
    It defines the expected input and output tensor shapes and types via its respective input
    and output pairs.
    """

    @abstractproperty
    def output_ports(self) -> List[TensorPort]:
        """
        Returns: Definition of the output port of this module.
        """
        pass

    @abstractproperty
    def input_ports(self) -> List[TensorPort]:
        """
        Returns: Definition of the input ports. The method `create` will receive arguments with shapes and types
        defined by this list, in an order corresponding to the order of this list.
        """
        pass

    @abstractproperty
    def input_tensors(self) -> Mapping[TensorPort, tf.Tensor]:
        """
        Returns: A mapping from input ports to the TF placeholders that correspond to them.
        """
        pass

    @abstractproperty
    def output_tensors(self) -> Mapping[TensorPort, tf.Tensor]:
        """
        Returns: A mapping from output ports to the TF placeholders that correspond to them.
        """
        pass

    def convert_to_feed_dict(self, mapping: Mapping[TensorPort, np.ndarray]) -> Mapping[tf.Tensor, np.ndarray]:
        result = {ph: mapping[port] for port, ph in self.input_tensors.items()}
        return result

    @abstractmethod
    def setup(self):
        """
        Sets up the module. This usually involves creating the actual tensorflow graph. It is expected
        to be called after the input module is set up. This means that shared resources, such as the vocab,
        are prepared already at this point.
        """
        pass


class SimpleModelModule(ModelModule):
    def __init__(self):
        """
        Creates a ModelModule and instantiates the TF graphs that implement the layer's function.
        """
        self.input_placeholders = [d.create_placeholder() for d in self.input_ports]

        self.outputs = None

    @abstractmethod
    def create(self, *input_tensors: tf.Tensor) -> Mapping[TensorPort, tf.Tensor]:
        """
        This function needs to be implemented in order to define how the module produces
        output and loss tensors from input tensors.
        Args:
            *input_tensors: a list of input tensors.

        Returns:
            mapping from output ports to their tensors.
        """
        pass

    def setup(self):
        self.input_placeholders = {d: d.create_placeholder() for d in self.input_ports}
        self.outputs = self.create(*[self.input_placeholders[port] for port in self.input_ports])

    @property
    def input_tensors(self) -> Mapping[TensorPort, tf.Tensor]:
        """
        Returns: The TF placeholders that correspond to the input ports.
        """
        return self.input_placeholders

    @property
    def output_tensors(self) -> Mapping[TensorPort, tf.Tensor]:
        """
        Returns: the output TF tensor that represents the prediction. This may be a matrix of candidate
        scores, or span markers, or actual symbol vectors generated. Should match the tensor type of `output_port`.
        """
        return self.outputs


class OutputModule(Module):
    """
    An output module takes the output (numpy) tensors of the model module and turns them into
    jack data structures.
    """

    @abstractproperty
    def input_ports(self) -> List[TensorPort]:
        """
        Returns: a port defines the input tensor to the output module.
        """
        pass

    @abstractmethod
    def __call__(self, inputs: List[Input], prediction: Mapping[TensorPort, np.ndarray]) -> List[Answer]:
        """
        Process the prediction tensor for a batch to produce a list of answers. The module has access
        to the original inputs.
        Args:
            inputs:
            prediction:

        Returns:

        """
        pass


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
                 shared_resources=None,
                 sess: tf.Session = None):
        self.shared_resources = shared_resources
        self.sess = sess or tf.Session()
        self.output_module = output_module
        self.model_module = model_module
        self.input_module = input_module
        assert self.input_module.output_ports == self.model_module.input_ports, \
            "Input Module outputs must match model module inputs"

        # fixme: this should test whether the output module inputs are a subset of the model-module outputs
        # assert self.model_module.output_port == self.output_module.input_port, \
        #     "Module model output must match output module inputs"

    def __call__(self, inputs: List[Input]) -> List[Answer]:
        """
        Reads all inputs (support and question), then returns an answer for each input.
        Args:
            inputs: a list of inputs.

        Returns: a list of answers.
        """
        batch = self.input_module(inputs)
        feed_dict = {self.model_module.input_tensors[port]: value for port, value in batch.items()}
        output_tensors = [self.model_module.output_tensors[port] for port in self.output_module.input_ports]
        predictions = self.sess.run(output_tensors, feed_dict=feed_dict)
        predictions_mapping = {self.output_module.input_ports[i]: prediction for i, prediction in
                               enumerate(predictions)}
        answers = self.output_module(inputs, predictions_mapping)
        return answers

    def setup(self, data: List[Tuple[Input, Answer]]):
        vocab = self.input_module.setup(data)
        self.model_module.setup(vocab)

    def train(self,
              training_set: List[Tuple[Input, Answer]],
              dev_set: List[Tuple[Input, Answer]] = None,
              test_set: List[Tuple[Input, Answer]] = None,
              **train_params):
        """
        This method trains the reader (and changes its state).
        Args:
            test_set: test set
            dev_set: dev set
            training_set: the training instances.
            **train_params: parameters to be sent to the training function `jtr.train.train`.

        Returns: None

        """
        train_port_mappings = self.input_module.training_generator(training_set)

        # note that this generator comprehension, not list comprehension
        # train_feed_dicts = (self.model_module.convert_to_feed_dict(m)
        #                    for m in train_port_mappings)

        train_feed_dicts = [self.model_module.convert_to_feed_dict(m) for m in train_port_mappings]

        hooks = [LossHook(1, 1)]
        args = {
            'loss': self.model_module.output_tensors[Ports.loss],
            'batches': train_feed_dicts,
            'hooks': hooks,
            'max_epochs': 100,
            **train_params
        }
        jtr_train.train(**args)

    def store(self):
        """
        Store module states and shared resources.
        """
        self.shared_resources.store()
        self.input_module.store()
        self.model_module.store()
        self.output_module.store()
