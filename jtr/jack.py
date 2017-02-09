"""
Here we define the basic interfaces of jtr. jtr readers consist of 3 layers, one that transform
jtr data structures into tensors, one that processes predicts the outputs and losses
using a tensorflow model into other tensors, and one that converts these tensors back to jtr data structures.
"""

from abc import abstractmethod, ABCMeta, abstractproperty
from typing import Mapping, Iterable, Tuple, Callable
import numpy as np
import tensorflow as tf
import jtr.train as jtr_train

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

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

    def __str__(self):
        return self.name + "_" + (self.shape_string or "[]")


class Ports:
    """
    This class groups input ports. Different modules can refer to these ports
    to define their input or output, respectively.
    """
    question = TensorPort(tf.int32, [None, None], "question",
                          "Represents questions using symbol vectors",
                          "[batch_size, max_num_question_tokens]")

    loss = TensorPort(tf.float32, [None],
                      "Represents loss on each instance in the batch",
                      "[batch_size]")


    # Support
    single_support = TensorPort(tf.int32, [None, None], "single_support",
                                "Represents instances with a single support document. "
                                "[batch_size, max_num_tokens]")
    multiple_support = TensorPort(tf.int32, [None, None, None], "multiple_support",
                                  "Represents instances with multiple support documents",
                                  "[batch_size, max_num_support, max_num_tokens]")


    # Candidates
    atomic_candidates = TensorPort(tf.int32, [None, None], "candidates",
                                   "Represents candidate choices using single symbols",
                                   "[batch_size, num_candidates]")

    candidate_scores = TensorPort(tf.float32, [None, None], "candidate_scores",
                    "Represents output scores for each candidate",
                    "[batch_size, num_candidates]")

    candidate_targets = TensorPort(tf.float32, [None, None], "candidate_targets",
                                   "Represents target (0/1) values for each candidate",
                                   "[batch_size, num_candidates]")

    candidate_index = TensorPort(tf.int32, [None], "answer_idx",
                                 "Represents answer as a single index",
                                 "[batch_size]")

    class Flat:
        """
         Number of questions in batch is Q, number of supports is S, number of answers is A, number of candidates is C.
        Typical input ports such as support, candidates, answers are defined together with individual mapping ports. This
        allows for more flexibility when numbers can vary between questions. Naming convention is to use suffix "_flat".
        """

        # Support
        support_to_question = TensorPort(tf.int32, [None], "support2question",
                                         "Represents mapping to question idx per support",
                                         "[S]")

        support = TensorPort(tf.int32, [None, None], "support_flat",
                                     "Represents instances with a single support document. "
                                     "[S, max_num_tokens]")

        # Candidates
        candidate_to_question = TensorPort(tf.int32, [None], "candidate2question",
                                           "Represents mapping to question idx per candidate",
                                           "[C]")

        atomic_candidates = TensorPort(tf.int32, [None], "candidates_flat",
                                       "Represents candidate choices using single symbols",
                                       "[C]")

        candidate_scores = TensorPort(tf.float32, [None], "candidate_scores_flat",
                                      "Represents output scores for each candidate",
                                      "[C]")

        seq_candidates = TensorPort(tf.int32, [None, None], "seq_candidates_flat",
                                    "Represents candidate choices using single symbols",
                                    "[C, max_num_tokens]")

        atomic_candidate_targets = TensorPort(tf.float32, [None], "candidate_targets_flat",
                                             "Represents groundtruth candidate labels, usually 1 or 0",
                                             "[C]")


        # Answers
        answer_to_question = TensorPort(tf.int32, [None], "answer2question",
                                        "Represents mapping to question idx per answer",
                                        "[A]")

        # -extractive
        seq_answer = TensorPort(tf.int32, [None, None], "answer_seq_flat",
                                "Represents answer as a sequence of symbols",
                                "[A, max_num_tokens]")

        answer_span = TensorPort(tf.int32, [None, 2], "answer_span_flat",
                                 "Represents answer as a (start, end) span",
                                 "[A, 2]")

        start_scores = TensorPort(tf.float32, [None, None], "start_scores",
                                  "Represents start scores for each support sequence",
                                  "[S, max_num_tokens]")

        end_scores = TensorPort(tf.float32, [None, None], "end_scores",
                                "Represents end scores for each support sequence",
                                "[S, max_num_tokens]")

        # -generative
        generative_symbol_scores = TensorPort(tf.int32, [None, None], "symbol_scores",
                                              "Represents symbol scores for each possible "
                                              "sequential answer given during training",
                                              "[A, max_num_tokens]")


        # MISC intermediate ports that might come in handy
        # -embeddings
        embedded_seq_candidates = TensorPort(tf.int32, [None, None, None], "embedded_seq_candidates_flat",
                                             "Represents the embedded sequential candidates",
                                             "[C, max_num_tokens, N]")

        embedded_candidates = TensorPort(tf.int32, [None, None], "embedded_candidates_flat",
                                         "Represents the embedded candidates",
                                         "[C, N]")

        embedded_support = TensorPort(tf.int32, [None, None, None], "embedded_support_flat",
                                      "Represents the embedded support",
                                      "[S, max_num_tokens, N]")

        embedded_question = TensorPort(tf.int32, [None, None, None], "embedded_question_flat",
                                   "Represents the embedded question",
                                   "[Q, max_num_question_tokens, N]")

        # -attention, ...


class InputModule(metaclass=ABCMeta):
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

    @abstractmethod
    def __call__(self, inputs: List[Input]) -> Mapping[TensorPort, np.ndarray]:
        """
        Converts a list of inputs into a single batch of tensors, consisting with the `output_ports` of this
        module.
        Args:
            inputs: a list of inputs, e.g., a QASetting

        Returns:
            A mapping from defined output ports to numpy tensors.

        """
        pass

    @abstractmethod
    def training_generator(self, training_set: List[Tuple[Input, List[Answer]]]) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        """
        Given a training set of input-answer pairs, this method produces an iterable/generator
        that when iterated over returns a sequence of batches. These batches map ports to tensors
        just as `__call__` does, using the `output_ports` of this object.
        Args:
            training_set: a set of pairs of input and answer.

        Returns: An iterable/generator that, on each pass through the data, produces a list of batches.
        """
        pass


class ModelModule(metaclass=ABCMeta):
    """
    A model module encapsulates two tensorflow trees (possibly overlapping): a tree representing
    the answer prediction (to be processed by the outout module) and a tree representing the loss.
    It defines the expected input and output tensor shapes and types via its respective input
    and output pairs.
    """

    def __call__(self, sess: tf.Session,
                 batch: Mapping[TensorPort, np.ndarray],
                 goal_ports: List[TensorPort]=None) -> Mapping[TensorPort, np.ndarray]:
        """
        Converts a list of inputs into a single batch of tensors, consisting with the `output_ports` of this
        module.
        Args:
            inputs: a list of instances (question, support, optional candidates)

        Returns:
            A mapping from ports to tensors.

        """
        if goal_ports is None:
            goals = self.output_tensors
        else:
            goals = [o for p, o in zip(self.output_ports, self.output_tensors) if p in goal_ports]

        feed_dict = self.convert_to_feed_dict(batch)
        outputs = sess.run(goals, feed_dict)

        return {p: outputs[goals.index(o)] for p, o in zip(self.output_ports, self.output_tensors) if o in goals}

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
    def input_tensors(self) -> List[tf.Tensor]:
        """
        Returns: The TF placeholders that correspond to the input ports.
        """
        pass

    @abstractproperty
    def output_tensors(self) -> List[tf.Tensor]:
        """
        Returns: the output TF tensor that represents the prediction. This may be a matrix of candidate
        scores, or span markers, or actual symbol vectors generated. Should match the tensor type of `output_port`.
        """
        pass

    def convert_to_feed_dict(self, mapping: Mapping[TensorPort, np.ndarray]) -> Mapping[tf.Tensor, np.ndarray]:
        return {input_tensor: mapping[port] for input_tensor, port in zip(self.input_tensors, self.input_ports)}


class SimpleModelModule(ModelModule):
    """
    A model module encapsulates two tensorflow trees (possibly overlapping): a tree representing
    the answer prediction (to be processed by the outout module) and a tree representing the loss.
    It defines the expected input and output tensor shapes and types via its respective input
    and output pairs.
    """

    def __init__(self, port2ph: Mapping[TensorPort, tf.Tensor]=None):
        """
        Args:
            port2ph: necessary when placeholders/tensors already exist for certain tensorports

        Returns:

        """
        if port2ph is not None:
            self.placeholders = [port2ph.get(d, d.create_placeholder()) for d in self.input_ports]
        else:
            self.placeholders = [d.create_placeholder() for d in self.input_ports]
        self._outputs = self.create(*self.placeholders)

    @abstractmethod
    def create(self, *tensors: tf.Tensor) -> List[tf.Tensor]:
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

    @property
    def input_tensors(self) -> List[tf.Tensor]:
        """
        Returns: The TF placeholders that correspond to the input ports.
        """
        return self.placeholders

    @property
    def output_tensors(self) -> List[tf.Tensor]:
        """
        Returns: the output TF tensor that represents the prediction. This may be a matrix of candidate
        scores, or span markers, or actual symbol vectors generated. Should match the tensor type of `output_port`.
        """
        return self._outputs


class StackedModelModule(ModelModule):
    """
    stacks ModelModules based on sequence of constructors that enable
    """

    def __init__(self, *module_contructors: Callable[[dict], ModelModule]):
        """

        Args:
            *module_contructors: functions from port -> placeholder/tensor dict to ModelModule

        Returns:

        """
        self._output_ports = []
        self._input_ports = []

        self._output_tensors = []
        self._input_tensors = []

        port2placeholder = dict()

        for module_contructor in module_contructors:
            module = module_contructor(port2placeholder)
            for p, t in zip(module.input_ports, module.input_tensors):
                if p not in port2placeholder:
                    # this input has to be provided
                    self._input_ports.append(p)
                    self._input_tensors.append(t)
                    port2placeholder[p] = t

            self._output_ports.extend(module.output_ports)
            self._output_tensors.extend(module.output_tensors)

            port2placeholder.update(zip(module.output_ports, module.output_tensors))


    @property
    def input_ports(self) -> List[TensorPort]:
        """
        Returns: a port defines the input tensor to the output module.
        """
        return self._input_ports

    @property
    def output_ports(self) -> List[TensorPort]:
        """
        Returns: a port defines the input tensor to the output module.
        """
        return self._output_ports

    @property
    def output_tensors(self) -> List[tf.Tensor]:
        return self._output_tensors

    @property
    def input_tensors(self) -> List[tf.Tensor]:
        return self._input_tensors


class OutputModule(metaclass=ABCMeta):
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
    def __call__(self, inputs: List[Input], model_outputs: Mapping[TensorPort, np.ndarray]) -> List[Answer]:
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
                 sess: tf.Session = None):
        self.sess = sess or tf.Session()
        self.output_module = output_module
        self.model_module = model_module
        self.input_module = input_module

        assert all(port in self.input_module.output_ports for port in self.model_module.input_ports), \
            "Input Module outputs must include model module inputs"

        assert all(port in self.model_module.output_ports for port in self.output_module.input_ports), \
            "Module model output must match output module inputs"

    def __call__(self, inputs: List[Input]) -> List[Answer]:
        """
        Reads all inputs (support and question), then returns an answer for each input.
        Args:
            inputs: a list of inputs.

        Returns: a list of answers, predictions made by the model.
        """
        batch = self.input_module(inputs)
        prediction = self.model_module(self.sess, batch)
        answers = self.output_module(inputs, prediction)
        return answers

    def run(self, inputs: List[Input], goals: List[TensorPort]) -> List[np.ndarray]:
        batch = self.input_module(inputs)
        return self.model_module(self.sess, batch, goals)
