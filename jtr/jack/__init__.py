"""
Here we define the basic interfaces of jtr. jtr readers consist of 3 layers, one that transform
jtr data structures into tensors, one that processes predicts the outputs and losses
using a tensorflow model into other tensors, and one that converts these tensors back to jtr data structures.
"""
import os
import pickle
from abc import abstractmethod, ABCMeta, abstractproperty
from typing import Mapping, Iterable, Tuple, Callable
import numpy as np
import shutil
import tensorflow as tf
import jtr.train as jtr_train

from jtr.jack.data_structures import *

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


class TensorPortWithDefault(TensorPort):
    def __init__(self, default_value, dtype, shape, name, doc_string=None, shape_string=None):
        self.default_value = default_value
        super().__init__(dtype, shape, name, doc_string, shape_string)

    def create_placeholder(self):
        """
        Convenience method that produces a constant of the type, value and shape defined by the port.
        Returns: a constant tensor of same type, shape and name. It can nevertheless be fed with external values
        as if it was a placeholder.
        """
        ph = tf.placeholder_with_default(self.default_value, self.shape, self.name)
        if ph.dtype != self.dtype:
            logging.warning("Placeholder %s with default of type %s created for TensorPort with type %s!" %
                            (self.name, ph.dtype, self.dtype))
        return ph


class Ports:
    """
    This class groups input ports. Different modules can refer to these ports
    to define their input or output, respectively.
    """

    loss = TensorPort(tf.float32, [None],
                      "Represents loss on each instance in the batch",
                      "[batch_size]")

    class Input:
        question = TensorPort(tf.int32, [None, None], "question",
                              "Represents questions using symbol vectors",
                              "[batch_size, max_num_question_tokens]")

        single_support = TensorPort(tf.int32, [None, None], "single_support",
                                    "Represents instances with a single support document. ",
                                    "[batch_size, max_num_tokens]")

        multiple_support = TensorPort(tf.int32, [None, None, None], "multiple_support",
                                      "Represents instances with multiple support documents",
                                      "[batch_size, max_num_support, max_num_tokens]")

        atomic_candidates = TensorPort(tf.int32, [None, None], "candidates",
                                       "Represents candidate choices using single symbols",
                                       "[batch_size, num_candidates]")

        keep_prob = TensorPortWithDefault(1.0, tf.float32, [], "keep_prob",
                                          "scalar representing keep probability when using dropout",
                                          "[]")

        is_eval = TensorPortWithDefault(True, tf.bool, [], "is_eval",
                                        "boolean that determines whether input is eval or training.",
                                        "[]")

    class Prediction:
        candidate_scores = TensorPort(tf.float32, [None, None], "candidate_scores",
                                      "Represents output scores for each candidate",
                                      "[batch_size, num_candidates]")

        candidate_index = TensorPort(tf.int32, [None], "candidate_idx",
                                     "Represents answer as a single index",
                                     "[batch_size]")

    class Targets:
        candidate_labels = TensorPort(tf.float32, [None, None], "candidate_targets",
                                      "Represents target (0/1) values for each candidate",
                                      "[batch_size, num_candidates]")


class FlatPorts:
    """
     Number of questions in batch is Q, number of supports is S, number of answers is A, number of candidates is C.
    Typical input ports such as support, candidates, answers are defined together with individual mapping ports. This
    allows for more flexibility when numbers can vary between questions. Naming convention is to use suffix "_flat".
    """

    class Input:
        support_to_question = TensorPort(tf.int32, [None], "support2question",
                                         "Represents mapping to question idx per support",
                                         "[S]")
        candidate_to_question = TensorPort(tf.int32, [None], "candidate2question",
                                           "Represents mapping to question idx per candidate",
                                           "[C]")
        answer2question = TensorPort(tf.int32, [None], "answer2question",
                                     "Represents mapping to question idx per answer",
                                     "[A]")

        support = TensorPort(tf.int32, [None, None], "support_flat",
                             "Represents instances with a single support document. "
                             "[S, max_num_tokens]")

        atomic_candidates = TensorPort(tf.int32, [None], "candidates_flat",
                                       "Represents candidate choices using single symbols",
                                       "[C]")

        seq_candidates = TensorPort(tf.int32, [None, None], "seq_candidates_flat",
                                    "Represents candidate choices using single symbols",
                                    "[C, max_num_tokens]")

        support_length = TensorPort(tf.int32, [None], "support_length_flat",
                                    "Represents length of support in batch",
                                    "[S]")

        question_length = TensorPort(tf.int32, [None], "question_length_flat",
                                     "Represents length of questions in batch",
                                     "[Q]")

    class Prediction:
        candidate_scores = TensorPort(tf.float32, [None], "candidate_scores_flat",
                                      "Represents output scores for each candidate",
                                      "[C]")

        candidate_idx = TensorPort(tf.float32, [None], "candidate_predictions_flat",
                                   "Represents groundtruth candidate labels, usually 1 or 0",
                                   "[C]")

        # extractive QA
        start_scores = TensorPort(tf.float32, [None, None], "start_scores_flat",
                                  "Represents start scores for each support sequence",
                                  "[S, max_num_tokens]")

        end_scores = TensorPort(tf.float32, [None, None], "end_scores_flat",
                                "Represents end scores for each support sequence",
                                "[S, max_num_tokens]")

        answer_span = TensorPort(tf.int32, [None, 2], "answer_span_prediction_flat",
                                 "Represents answer as a (start, end) span", "[A, 2]")

        # generative QA
        generative_symbol_scores = TensorPort(tf.int32, [None, None, None], "symbol_scores",
                                              "Represents symbol scores for each possible "
                                              "sequential answer given during training",
                                              "[A, max_num_tokens, vocab_len]")

        generative_symbols = TensorPort(tf.int32, [None, None], "symbol_prediction",
                                        "Represents symbol sequence for each possible "
                                        "answer predicted by the model",
                                        "[A, max_num_tokens]")

    class Target:
        candidate_idx = TensorPort(tf.float32, [None], "candidate_targets_flat",
                                   "Represents groundtruth candidate labels, usually 1 or 0",
                                   "[C]")

        answer_span = TensorPort(tf.int32, [None, 2], "answer_span_target_flat",
                                 "Represents answer as a (start, end) span", "[A, 2]")

        seq_answer = TensorPort(tf.int32, [None, None], "answer_seq_target_flat",
                                "Represents answer as a sequence of symbols",
                                "[A, max_num_tokens]")

        generative_symbols = TensorPort(tf.int32, [None, None], "symbol_targets",
                                        "Represents symbol scores for each possible "
                                        "sequential answer given during training",
                                        "[A, max_num_tokens]")

    class Misc:
        # MISC intermediate ports that might come in handy
        # -embeddings
        embedded_seq_candidates = TensorPort(tf.float32, [None, None, None], "embedded_seq_candidates_flat",
                                             "Represents the embedded sequential candidates",
                                             "[C, max_num_tokens, N]")

        embedded_candidates = TensorPort(tf.float32, [None, None], "embedded_candidates_flat",
                                         "Represents the embedded candidates",
                                         "[C, N]")

        embedded_support = TensorPort(tf.float32, [None, None, None], "embedded_support_flat",
                                      "Represents the embedded support",
                                      "[S, max_num_tokens, N]")

        embedded_question = TensorPort(tf.float32, [None, None, None], "embedded_question_flat",
                                       "Represents the embedded question",
                                       "[Q, max_num_question_tokens, N]")
        # -attention, ...


# alias: shared resources can be any kind of pickable object
SharedResources = object


class SharedVocabAndConfig(SharedResources):
    """
    A class to provide and store a vocab shared across some of the reader modules.
    """

    def __init__(self, vocab, config=None):
        self.config = config
        self.vocab = vocab


class InputModule:
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
    def training_ports(self) -> List[TensorPort]:
        """
        Defines what types of tensor is used to represent the target solution during training.
        """
        pass

    @abstractmethod
    def __call__(self, inputs: List[Question]) -> Mapping[TensorPort, np.ndarray]:
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
    def dataset_generator(self,
                          dataset: List[Tuple[Question, List[Answer]]],
                          is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        """
        Given a training set of input-answer pairs, this method produces an iterable/generator
        that when iterated over returns a sequence of batches. These batches map ports to tensors
        just as `__call__` does, using the `output_ports` of this object.
        Args:
            dataset: a set of pairs of input and answer.

        Returns: An iterable/generator that, on each pass through the data, produces a list of batches.
        """
        pass

    @abstractmethod
    def setup_from_data(self, data: List[Tuple[Question, List[Answer]]]) -> SharedResources:
        """
        Args:
            data: a set of pairs of input and answer.

        Returns: vocab
        """
        pass

    @abstractmethod
    def setup(self, shared_resources: SharedResources):
        """
        Args:
            data: a set of pairs of input and answer.
        """
        pass

    def store(self, path):
        """
        Store the state of this module. Default is that there is no state, so nothing to store.
        """
        pass

    def load(self, path):
        """
        Load the state of this module. Default is that there is no state, so nothing to load.
        """
        pass


class ModelModule:
    """
    A model module encapsulates two tensorflow trees (possibly overlapping): a tree representing
    the answer prediction (to be processed by the outout module) and a tree representing the loss.
    It defines the expected input and output tensor shapes and types via its respective input
    and output pairs.
    """

    def __call__(self, sess: tf.Session,
                 batch: Mapping[TensorPort, np.ndarray],
                 goal_ports: List[TensorPort] = list()) -> Mapping[TensorPort, np.ndarray]:
        """
        Converts a list of inputs into a single batch of tensors, consisting with the `output_ports` of this
        module.
        Args:
            inputs: a list of instances (question, support, optional candidates)

        Returns:
            A mapping from ports to tensors.

        """
        goal_ports = goal_ports or self.output_ports

        feed_dict = self.convert_to_feed_dict(batch)
        outputs = sess.run([self.tensors[p] for p in goal_ports if p in self.output_ports], feed_dict)

        ret = dict(zip(filter(lambda p: p in self.output_ports, goal_ports), outputs))
        for p in goal_ports:
            if p not in ret and p in batch:
                ret[p] = batch[p]

        return ret

    @abstractproperty
    def output_ports(self) -> List[TensorPort]:
        """
        Returns: Definition of the output ports of this module.
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
    def training_input_ports(self) -> List[TensorPort]:
        """
        Returns: A mapping from input target ports to the TF placeholders that correspond to them.
        """
        pass

    @abstractproperty
    def training_output_ports(self) -> List[TensorPort]:
        """
        Returns: Definition of the training ports of this module.
        """
        pass

    @abstractproperty
    def placeholders(self) -> Mapping[TensorPort, tf.Tensor]:
        """
        Returns: A mapping from ports to the TF placeholders that correspond to them.
        """
        pass

    @abstractproperty
    def tensors(self) -> Mapping[TensorPort, tf.Tensor]:
        """
        Returns: A mapping from ports to the TF tensors that correspond to them.
        """
        pass

    def convert_to_feed_dict(self, mapping: Mapping[TensorPort, np.ndarray]) -> Mapping[tf.Tensor, np.ndarray]:
        result = {ph: mapping[port] for port, ph in self.placeholders.items() if port in mapping}
        return result

    @abstractmethod
    def setup(self, shared_resources: SharedResources, is_training=True):
        """
        Sets up the module. This usually involves creating the actual tensorflow graph. It is expected
        to be called after the input module is set up. This means that shared resources, such as the vocab,
        are prepared already at this point.
        """
        pass

    def store(self, sess, path):
        """
        Store the state of this module. Default is that there is no state, so nothing to store.
        """
        pass

    def load(self, sess, path):
        """
        Load the state of this module. Default is that there is no state, so nothing to load.
        """
        pass

    @abstractproperty
    def train_variables(self) -> List[tf.Variable]:
        """ Returns: A list of training variables """

    @abstractproperty
    def variables(self) -> List[tf.Variable]:
        """ Returns: A list of variables """


class SimpleModelModule(ModelModule):
    @abstractmethod
    def create_output(self, shared_resources: SharedResources, *input_tensors: tf.Tensor) -> Mapping[
        TensorPort, tf.Tensor]:
        """
        This function needs to be implemented in order to define how the module produces
        output and loss tensors from input tensors.
        Args:
            *input_tensors: a list of input tensors.

        Returns:
            mapping from output ports to their tensors.
        """
        pass

    @abstractmethod
    def create_training_output(self, shared_resources: SharedResources, *target_input_tensors: tf.Tensor) -> Mapping[
        TensorPort, tf.Tensor]:
        """
        This function needs to be implemented in order to define how the module produces
        output and loss tensors from input tensors.
        Args:
            *input_tensors: a list of input tensors.

        Returns:
            mapping from output ports to their tensors.
        """
        pass

    def setup(self, shared_resources: SharedResources, is_training=True):
        old_train_variables = tf.trainable_variables()
        old_variables = tf.global_variables()
        self._tensors = {d: d.create_placeholder() for d in self.input_ports}
        self._placeholders = dict(self._tensors)
        output_tensors = self.create_output(shared_resources, *[self._tensors[port] for port in self.input_ports])
        self._tensors.update(zip(self.output_ports, output_tensors))
        if is_training:
            self._placeholders.update((p, p.create_placeholder()) for p in self.training_input_ports
                                      if p not in self._placeholders and p not in self._tensors)
            self._tensors.update(self._placeholders)
            input_target_tensors = {p: self._tensors.get(p, None) for p in self.training_input_ports}
            training_output_tensors = self.create_training_output(shared_resources, *[input_target_tensors[port]
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
    def train_variables(self) -> List[tf.Tensor]:
        """ Returns: A list of training variables """
        return self._training_variables

    @property
    def variables(self) -> List[tf.Tensor]:
        """ Returns: A list of variables """
        return self._variables


class OutputModule:
    """
    An output module takes the output (numpy) tensors of the model module and turns them into
    jack data structures.
    """

    @abstractproperty
    def input_ports(self) -> List[TensorPort]:
        """
        Returns: correspond to a subset of output ports of model module.
        """
        pass

    @abstractmethod
    def __call__(self, inputs: List[Question], *tensor_inputs: np.ndarray) -> List[Answer]:
        """
        Process the prediction tensor for a batch to produce a list of answers. The module has access
        to the original inputs.
        Args:
            inputs:
            prediction:

        Returns:

        """
        pass

    @abstractmethod
    def setup(self, shared_resources: SharedResources):
        """
        Args:
            shared_resources: sets up this module with shared resources
        """

    def store(self, path):
        """
        Store the state of this module. Default is that there is no state, so nothing to store.
        """
        pass

    def load(self, path):
        """
        Load the state of this module. Default is that there is no state, so nothing to load.
        """
        pass


class JTReader:
    """
    A Reader reads inputs consisting of questions, supports and possibly candidates, and produces answers.
    It consists of three layers: input to tensor (input_module), tensor to tensor (model_module), and tensor to answer
    (output_model). These layers are called in-turn on a given input (list).
    """

    def __init__(self,
                 input_module: InputModule,
                 model_module: ModelModule,
                 output_module: OutputModule,
                 sess: tf.Session = None,
                 is_train: bool = True,
                 shared_resources=None):
        self.shared_resources = shared_resources
        self.sess = sess
        self.output_module = output_module
        self.model_module = model_module
        self.input_module = input_module
        self.is_train = is_train

        if self.sess is None:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=sess_config)

        assert all(port in self.input_module.output_ports for port in self.model_module.input_ports), \
            "Input Module outputs must include model module inputs"

        assert all(port in self.input_module.training_ports or port in self.model_module.output_ports or
                   port in self.input_module.output_ports for port in self.model_module.training_input_ports), \
            "Input Module (training) outputs and model module outputs must include model module training inputs"

        assert all(port in self.model_module.output_ports or port in self.input_module.output_ports
                   for port in self.output_module.input_ports), \
            "Module model output must match output module inputs"

    def __call__(self, inputs: List[Question]) -> List[Answer]:
        """
        Reads all inputs (support and question), then returns an answer for each input.
        Args:
            inputs: a list of inputs.

        Returns: a list of answers.
        """
        batch = self.input_module(inputs)
        output_module_input = self.model_module(self.sess, batch, self.output_module.input_ports)
        answers = self.output_module(inputs, *[output_module_input[p] for p in self.output_module.input_ports])
        return answers

    def train(self, optim,
              training_set: List[Tuple[Question, Answer]],
              max_epochs=10, hooks=[],
              l2=0.0, clip=None, clip_op=tf.clip_by_value):
        """
        This method trains the reader (and changes its state).
        Args:
            test_set: test set
            dev_set: dev set
            training_set: the training instances.
            **train_params: parameters to be sent to the training function `jtr.train.train`.

        Returns: None

        """
        assert self.is_train, "Reader has to be created for with is_train=True for training."

        logging.info("Setting up data and model...")
        # First setup shared resources, e.g., vocabulary. This depends on the input module.
        self.setup_from_data(training_set)

        batches = self.input_module.dataset_generator(training_set, is_eval=False)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        loss = self.model_module.tensors[Ports.loss]

        if l2 != 0.0:
            loss += \
                tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2

        if clip is not None:
            gradients = optim.compute_gradients(loss)
            if clip_op == tf.clip_by_value:
                gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                             for grad, var in gradients]
            elif clip_op == tf.clip_by_norm:
                gradients = [(tf.clip_by_norm(grad, clip), var)
                             for grad, var in gradients]
            min_op = optim.apply_gradients(gradients)
        else:
            min_op = optim.minimize(loss)

        # initialize non model variables like learning rate, optim vars ...
        self.sess.run([v.initializer for v in tf.global_variables() if v not in self.model_module.variables])

        logging.info("Start training...")
        for i in range(1, max_epochs + 1):
            for j, batch in enumerate(batches):
                feed_dict = self.model_module.convert_to_feed_dict(batch)
                _, current_loss = self.sess.run([min_op, loss], feed_dict=feed_dict)

                for hook in hooks:
                    hook.at_iteration_end(i, current_loss)

            # calling post-epoch hooks
            for hook in hooks:
                hook.at_epoch_end(i)

    def setup_from_data(self, data: List[Tuple[Question, Answer]]):
        """
        Overrides shared-resources
        Args:
            data: for instance the training dataset

        Returns:

        """
        self.shared_resources = self.input_module.setup_from_data(data)
        self.model_module.setup(self.shared_resources, self.is_train)
        self.output_module.setup(self.shared_resources)
        self.sess.run([v.initializer for v in self.model_module.variables])

    def setup_from_file(self, dir):
        with open(os.path.join(dir, "shared_resources"), 'rb') as f:
            self.shared_resources = pickle.load(f)
        self.input_module.setup(self.shared_resources)
        self.input_module.load(os.path.join(dir, "input_module"))
        self.model_module.setup(self.shared_resources, self.is_train)
        self.sess.run([v.initializer for v in self.model_module.variables])
        self.model_module.load(self.sess, os.path.join(dir, "model_module"))
        self.output_module.setup(self.shared_resources)
        self.output_module.load(os.path.join(dir, "output_module"))

    def load(self, dir):
        # assumes that everything is setup perfectly
        self.input_module.load(os.path.join(dir, "input_module"))
        self.model_module.load(self.sess, os.path.join(dir, "model_module"))
        self.output_module.load(os.path.join(dir, "output_module"))

    def store(self, dir):
        """
        Store module states and shared resources.
        """
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        with open(os.path.join(dir, "shared_resources"), "wb") as f:
            pickle.dump(self.shared_resources, f)
        self.input_module.store(os.path.join(dir, "input_module"))
        self.model_module.store(self.sess, os.path.join(dir, "model_module"))
        self.output_module.store(os.path.join(dir, "output_module"))
