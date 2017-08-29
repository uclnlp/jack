# -*- coding: utf-8 -*-

"""
Here we define the basic interfaces of jtr. jtr readers consist of 3 layers, one that transform
jtr data structures into tensors, one that processes predicts the outputs and losses
using a TensorFlow model into other tensors, and one that converts these tensors back to jtr data structures.
"""

import logging
import os
import pickle
import random
import shutil
import sys
from abc import abstractmethod
from typing import Mapping, Iterable, Generic, TypeVar, Optional

import numpy as np
import tensorflow as tf

from jtr.data_structures import *
from jtr.util.batch import shuffle_and_batch
from jtr.util.vocab import Vocab

logger = logging.getLogger(__name__)


_rng = random.Random(1234)


class TensorPort:
    """
    A TensorPort defines an input or output tensor for a ModelModule. This subsumes a
    shape of the tensor, and its data type.
    """

    def __init__(self, dtype, shape, name, doc_string=None, shape_string=None):
        """
        Create a new TensorPort.
        
        Args:
            dtype: the (TF) data type of the port.
            shape: the shape of the tensor.
            name: the name of this port (should be a valid TF name)
            doc_string: a documentation string associated with this port
            shape_string: a string of the form [size_1,size_2,size_3] where size_i is a text describing the
                size of the tensor's dimension i (such as "number of batches").
        """
        self.dtype = dtype
        self.shape = shape
        self.name = name
        self.__doc__ = doc_string
        self.shape_string = shape_string

    def create_placeholder(self):
        """
        Convenience method that produces a placeholder of the type and shape defined by the port.
        
        Returns: a placeholder of same type, shape and name.
        """
        return tf.placeholder(self.dtype, self.shape, self.name)

    def __gt__(self, port):
        return self.name > port.name


    def __repr__(self):

        return "<TensorPort (%s)>" % self.name


class TensorPortWithDefault(TensorPort):
    """
    TensorPort that also defines a default value.
    """

    def __init__(self, default_value, dtype, shape, name, doc_string=None, shape_string=None):
        self.default_value = default_value
        super().__init__(dtype, shape, name, doc_string=doc_string, shape_string=shape_string)

    def create_placeholder(self):
        """
        Convenience method that produces a constant of the type, value and shape defined by the port.
        Returns: a constant tensor of same type, shape and name. It can nevertheless be fed with external values
        as if it was a placeholder.
        """
        ph = tf.placeholder_with_default(self.default_value, self.shape, self.name)
        if ph.dtype != self.dtype:
            logger.warning(
                "Placeholder {} with default of type {} created for TensorPort with type {}!".format(self.name,
                                                                                                     ph.dtype,
                                                                                                     self.dtype))
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

        multiple_support = TensorPort(tf.int32, [None, None, None], "multiple_support",
                                      ("Represents instances with multiple support documents",
                                       " or single instances with extra dimension set to 1"),
                                      "[batch_size, max_num_support, max_num_tokens]")

        atomic_candidates = TensorPort(tf.int32, [None, None], "candidates",
                                       ("Represents candidate choices using single symbols. ",
                                        "This could be a list of entities from global entities ",
                                        "for example atomic_candidates = [e1, e7, e83] from ",
                                        "global_entities = [e1, e2, e3, ..., eN-1, eN"),
                                       "[batch_size, num_candidates]")

        sample_id = TensorPort(tf.int32, [None], "sample_id",
                               "Maps this sample to the index in the input text data",
                               "[batch_size]")

        support_length = TensorPort(tf.int32, [None, None], "support_length",
                                    "Represents length of supports in each support in batch",
                                    "[batch_size, num_supports]")

        question_length = TensorPort(tf.int32, [None], "question_length",
                                     "Represents length of questions in batch",
                                     "[Q]")

    class Prediction:
        logits = TensorPort(tf.float32, [None, None], "candidate_scores",
                            "Represents output scores for each candidate",
                            "[batch_size, num_candidates]")

        candidate_index = TensorPort(tf.float32, [None], "candidate_idx",
                                     "Represents answer as a single index",
                                     "[batch_size]")

    class Target:
        candidate_1hot = TensorPort(tf.float32, [None, None], "candidate_targets",
                                    "Represents target (0/1) values for each candidate",
                                    "[batch_size, num_candidates]")

        target_index = TensorPort(tf.int32, [None], "target_index",
                                  ("Represents symbol id of target candidate. ",
                                   "This can either be an index into a full list of candidates,",
                                   " which is fixed, or an index into a partial list of ",
                                   "candidates, for example a list of potential entities ",
                                   "from a list of many candiadtes"),
                                  "[batch_size]")


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
                                        "answer target_indexpredicted by the model",
                                        "[A, max_num_tokens]")

    class Target:
        candidate_idx = TensorPort(tf.int32, [None], "candidate_targets_flat",
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


class SharedResources():
    """
    A class to provide and store generally shared resources, such as vocabularies,
    across the reader sub-modules.
    """

    def __init__(self, vocab: Vocab = None, config: dict = None):
        """
        Several shared resources are initialised here, even if no arguments
        are passed when calling __init__.
        The instantiated objects will be filled by the InputModule.
        - self.config holds hyperparameter values and general configuration
            parameters.
        - self.vocab serves as default Vocabulary object.
        - self.answer_vocab is by default the same as self.vocab. However,
            this attribute can be changed by the InputModule, e.g. by setting
            sepvocab=True when calling the setup_from_data() of the InputModule.
        """
        self.config = config or dict()
        self.vocab = vocab

    def store(self, path):
        """
        Saves all attributes of this object.
        
        Args:
            path: path to save shared resources
        """
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """
        Loads this (potentially empty) resource from path (all object attributes).
        Args:
            path: path to shared resources
        """
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.__dict__.update(pickle.load(f))


class InputModule:
    """
    An input module processes inputs and turns them into tensors to be processed by the model module. Note that all
    setting up should be done in the setup method, NOT in the constructor. Only use the constructor to hand over
    external variables/states, like `SharedResources`.
    """

    @abstractmethod
    def setup(self):
        """Sets up the module (if needs setup after loading shared resources for instance) assuming shared resources
        are fully setup, usually called after loading and after `setup_from_data` as well."""
        pass

    @abstractmethod
    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]], dataset_name=None, identifier=None):
        """
        Sets up the module based on input data. This usually involves setting up vocabularies and other resources. This
        should and is only called before training, not before loading a saved model.
        
        Args:
            data: a set of pairs of input and answer.
        """
        raise NotImplementedError

    @abstractmethod
    def output_ports(self) -> List[TensorPort]:
        """
        Defines what types of tensors the output module produces in each batch.
        Returns: a list of tensor ports that correspond to the tensor ports in the mapping
        produced by `__call__`. The `batch_generator` method will return bindings for these
        ports and the ones in `training_ports`.
        """
        raise NotImplementedError

    @abstractmethod
    def training_ports(self) -> List[TensorPort]:
        """
        Defines what types of tensor are provided in addition to `output_ports` during training
        in the `batch_generator` function. Typically these will be ports that describe
        the target solution at training time.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, qa_settings: List[QASetting]) -> Mapping[TensorPort, np.ndarray]:
        """
        Converts a list of inputs into a single batch of tensors, consistent with the `output_ports` of this
        module.
        Args:
            qa_settings: a list of instances (question, support, optional candidates)

        Returns:
            A mapping from ports to tensors.

        """
        raise NotImplementedError

    @abstractmethod
    def batch_generator(self, dataset: Iterable[Tuple[QASetting, List[Answer]]],
                        is_eval: bool, dataset_name=None,
                        identifier=None) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        """
        Given a training set of input-answer pairs, this method produces an iterable/generator
        that when iterated over returns a sequence of batches. These batches map ports to tensors
        just as `__call__` does, but provides additional bindings for the `training_ports` ports in
        case `is_eval` is `False`.
        
        Args:
            dataset: a set of pairs of input and answer.
            is_eval: is this dataset generated for evaluation only (not training).

        Returns: An iterable/generator that, on each pass through the data, produces a list of batches.
        """
        raise NotImplementedError

    def store(self, path):
        """Store the state of this module. Default is that there is no state, so nothing to store."""
        pass

    def load(self, path):
        """Load the state of this module. Default is that there is no state, so nothing to load."""
        pass


AnnotationType = TypeVar('AnnotationType')
class OnlineInputModule(InputModule, Generic[AnnotationType]):
    """InputModule that preprocesses instances on the fly.

    It provides implementations for `create_batch()` and `__call__()` and
    introduces two abstract methods:
    - `preprocess()`: Converts a list of instances to annotations.
    - `create_batch()`: Converts a list of annotations to a batch.

    Both of these methods are parameterized by `AnnotationType`. In the simplest
    case, this could be a `dict`, but you could also define a separate class
    for your annotation, in order to get stronger typing.
    """

    @abstractmethod
    def preprocess(self, questions: List[QASetting],
                   answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) \
            -> List[AnnotationType]:
        """
        Preprocesses a list of samples, returning a list of annotations.
        Batches of these annotation objects are then passed to the
        the `create_batch` method.
        Args:
            questions: The list of instances to preprocess
            answers: (Optional) answers associated with the instances
            is_eval: Whether this preprocessing is done for evaluation data

        Returns:
            List of annotations of the instances.
        """

        raise NotImplementedError

    @abstractmethod
    def create_batch(self, annotations: List[AnnotationType],
                     is_eval: bool, with_answers: bool) \
            -> Mapping[TensorPort, np.ndarray]:
        """
        Creates a batch from a list of preprocessed questions, given by
        a list of annotations as returned by `preprocess_instance`.
        Args:
            annotations: a list of annotations to be included in the batch
            is_eval: whether the method is called for evaluation data
            with_answers: whether answers are included in the annotations

        Returns:
            A mapping from ports to numpy arrays.
        """

        raise NotImplementedError

    def batch_annotations(self, annotations: List[AnnotationType],
                          is_eval: bool):
        """Optionally shuffles and batches annotations.

        By default, all annotations are shuffled (if self.shuffle(is_eval) and
        then batched. Override this method if you want to customize the
        batching, e.g., to do stratified sampling, sampling with replacement,
        etc.

        Args:
            - annotations: List of annotations to shuffle & batch.
            - is_eval: Whether batches are generated for evaluation.

        Returns: Batch iterator
        """
        rng = _rng if self.shuffle(is_eval) else None
        return shuffle_and_batch(annotations, self.batch_size, rng)

    @property
    def batch_size(self):
        return 32

    def shuffle(self, is_eval):
        """Whether to shuffle the dataset in batch_annotations()."""
        return not is_eval

    def __call__(self, qa_settings: List[QASetting]) \
            -> Mapping[TensorPort, np.ndarray]:
        """Preprocesses all qa_settings, returns a single batch with all instances."""

        annotations = self.preprocess(qa_settings, answers=None, is_eval=True)
        return self.create_batch(annotations, is_eval=True, with_answers=False)

    def batch_generator(self,
                        dataset: Iterable[Tuple[QASetting, List[Answer]]],
                        is_eval: bool,
                        dataset_name=None,
                        identifier=None) \
            -> Iterable[Mapping[TensorPort, np.ndarray]]:
        """Preprocesses all instances, batches & shuffles them and generates batches."""

        questions, answers = zip(*dataset)
        annotations = self.preprocess(questions, answers, is_eval=is_eval)

        def make_generator():
            for annotation_batch in self.batch_annotations(annotations, is_eval):
                yield self.create_batch(annotation_batch, is_eval, True)

        return GeneratorWithRestart(make_generator)


class ModelModule:
    """
    A model module encapsulates two tensorflow trees (possibly overlapping): a tree representing
    the answer prediction (to be processed by the outout module) and a tree representing the loss.
    It defines the expected input and output tensor shapes and types via its respective input
    and output pairs.
    """

    def __call__(self, session: tf.Session,
                 batch: Mapping[TensorPort, np.ndarray],
                 goal_ports: List[TensorPort] = None) -> Mapping[TensorPort, np.ndarray]:
        """
        Runs a batch represented by a mapping from tensorports to numpy arrays and returns value for specified
        goal ports.
        Args:
            session: the tf session to use
            batch: mapping from ports to values
            goal_ports: optional output ports, defaults to output_ports of this module will be returned

        Returns:
            A mapping from goal ports to tensors.

        """
        goal_ports = goal_ports or self.output_ports

        feed_dict = self.convert_to_feed_dict(batch)
        outputs = session.run([self.tensors[p] for p in goal_ports if p in self.output_ports], feed_dict)

        ret = dict(zip(filter(lambda p: p in self.output_ports, goal_ports), outputs))
        for p in goal_ports:
            if p not in ret and p in batch:
                ret[p] = batch[p]

        return ret

    @abstractmethod
    def output_ports(self) -> Sequence[TensorPort]:
        """
        Returns: Definition of the output ports of this module.
        """
        raise NotImplementedError

    @abstractmethod
    def input_ports(self) -> Sequence[TensorPort]:
        """
        Returns: Definition of the input ports.
        """
        raise NotImplementedError

    @abstractmethod
    def training_input_ports(self) -> Sequence[TensorPort]:
        """
        Returns: Definition of the input ports necessary to create the training output ports, i.e., they do not have
        to be provided during eval and they can include output ports of this module.
        """
        raise NotImplementedError

    @abstractmethod
    def training_output_ports(self) -> Sequence[TensorPort]:
        """
        Returns: Definition of the output ports provided during training for this module.
        """
        raise NotImplementedError

    @abstractmethod
    def placeholders(self) -> Mapping[TensorPort, tf.Tensor]:
        """
        Returns: A mapping from ports to the TF placeholders that correspond to them.
        """
        raise NotImplementedError

    @abstractmethod
    def tensors(self) -> Mapping[TensorPort, tf.Tensor]:
        """
        Returns: A mapping from ports to the TF tensors that correspond to them.
        """
        raise NotImplementedError

    def convert_to_feed_dict(self, mapping: Mapping[TensorPort, np.ndarray]) -> Mapping[tf.Tensor, np.ndarray]:
        result = {ph: mapping[port] for port, ph in self.placeholders.items() if port in mapping}
        return result

    @abstractmethod
    def setup(self, is_training=True):
        """
        Sets up the module. This usually involves creating the actual tensorflow graph. It is expected
        to be called after the input module is set up and shared resources, such as the vocab, config, etc.,
        are prepared already at this point.
        """
        raise NotImplementedError

    def store(self, sess, path):
        """
        Store the state of this module. Default is that there is no state, so nothing to store.
        """
        raise NotImplementedError

    def load(self, sess, path):
        """
        Load the state of this module. Default is that there is no state, so nothing to load.
        """
        raise NotImplementedError

    @abstractmethod
    def train_variables(self) -> Sequence[tf.Variable]:
        """ Returns: A list of training variables """
        raise NotImplementedError

    @property
    @abstractmethod
    def variables(self) -> Sequence[tf.Variable]:
        """ Returns: A list of variables """
        raise NotImplementedError


class SimpleModelModule(ModelModule):
    """
    This class simplifies the implementation of ModelModules by requiring to implement a small set of methods that
    produce the TF graphs to create predictions and the training outputs, and define the ports.
    """

    def __init__(self, shared_resources: SharedResources):
        self.shared_resources = shared_resources

    @abstractmethod
    def create_output(self, shared_resources: SharedResources,
                      *input_tensors: tf.Tensor) -> Sequence[tf.Tensor]:
        """
        This function needs to be implemented in order to define how the module produces
        output from input tensors corresponding to `input_ports`.
        
        Args:
            *input_tensors: a list of input tensors.

        Returns:
            mapping from defined output ports to their tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def create_training_output(self, shared_resources: SharedResources,
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
        raise NotImplementedError

    def setup(self, is_training=True):
        old_train_variables = tf.trainable_variables()
        old_variables = tf.global_variables()
        self._tensors = {d: d.create_placeholder() for d in self.input_ports}
        self._placeholders = dict(self._tensors)
        output_tensors = self.create_output(self.shared_resources, *[self._tensors[port] for port in self.input_ports])
        self._tensors.update(zip(self.output_ports, output_tensors))
        if is_training:
            self._placeholders.update((p, p.create_placeholder()) for p in self.training_input_ports
                                      if p not in self._placeholders and p not in self._tensors)
            self._tensors.update(self._placeholders)
            input_target_tensors = {p: self._tensors.get(p, None) for p in self.training_input_ports}
            training_output_tensors = self.create_training_output(self.shared_resources, *[input_target_tensors[port]
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


class OutputModule:
    """
    An output module takes the output (numpy) tensors of the model module and turns them into
    jack data structures.
    """

    @abstractmethod
    def input_ports(self) -> Sequence[TensorPort]:
        """Returns: correspond to a subset of output ports of model module."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, inputs: Sequence[QASetting], *tensor_inputs: np.ndarray) -> Sequence[Answer]:
        """
        Process the tensors corresponding to the defined `input_ports` for a batch to produce a list of answers.
        The module has access to the original inputs.
        Args:
            inputs:
            prediction:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def setup(self):
        pass

    def store(self, path):
        """Store the state of this module. Default is that there is no state, so nothing to store."""
        pass

    def load(self, path):
        """Load the state of this module. Default is that there is no state, so nothing to load."""
        pass


class JTReader:
    """
    A Reader reads inputs consisting of questions, supports and possibly candidates, and produces answers.
    It consists of three layers: input to tensor (input_module), tensor to tensor (model_module), and tensor to answer
    (output_model). These layers are called in-turn on a given input (list).
    """

    def __init__(self,
                 shared_resources: SharedResources,
                 input_module: InputModule,
                 model_module: ModelModule,
                 output_module: OutputModule,
                 session: tf.Session = None,
                 is_train: bool = True):
        self.shared_resources = shared_resources
        self.session = session
        self.output_module = output_module
        self.model_module = model_module
        self.input_module = input_module
        self.is_train = is_train

        if self.session is None:
            session_config = tf.ConfigProto(allow_soft_placement=True)
            session_config.gpu_options.allow_growth = True
            self.session = tf.Session(config=session_config)

        assert all(port in self.input_module.output_ports for port in self.model_module.input_ports), \
            "Input Module outputs must include model module inputs"

        assert all(port in self.input_module.training_ports or port in self.model_module.output_ports or
                   port in self.input_module.output_ports for port in self.model_module.training_input_ports), \
            "Input Module (training) outputs and model module outputs must include model module training inputs"

        assert all(port in self.model_module.output_ports or port in self.input_module.output_ports
                   for port in self.output_module.input_ports), \
            "Module model output must match output module inputs"

    def __call__(self, inputs: Sequence[QASetting]) -> Sequence[Answer]:
        """
        Answers a list of question settings
        Args:
            inputs: a list of inputs.

        Returns:
            predicted outputs/answers to a given (labeled) dataset
        """
        batch = self.input_module(inputs)
        output_module_input = self.model_module(self.session, batch, self.output_module.input_ports)
        answers = self.output_module(inputs, *[output_module_input[p] for p in self.output_module.input_ports])
        return answers

    def process_outputs(self, dataset: Sequence[Tuple[QASetting, Answer]], batch_size: int, debug=False):
        """
        Similar to the call method, only that it works on a labeled dataset and applies batching. However, assumes
        that batches in input_module.batch_generator are processed in order and do not get shuffled during with
        flag is_eval set to true.
        
        Args:
            dataset:
            batch_size: note this information is needed here, but does not set the batch_size the model is using.
            This has to happen during setup/configuration.
            debug: if true, logging counter

        Returns:
            predicted outputs/answers to a given (labeled) dataset
        """
        logger.debug("Setting up batches...")
        batches = self.input_module.batch_generator(dataset, is_eval=True)
        answers = list()
        logger.debug("Start answering...")
        for j, batch in enumerate(batches):
            output_module_input = self.model_module(self.session, batch, self.output_module.input_ports)
            inputs = [x for x, _ in dataset[j * batch_size:(j + 1) * batch_size]]
            answers.extend(
                self.output_module(inputs, *[output_module_input[p] for p in self.output_module.input_ports]))
            if debug:
                logger.debug("{}/{} examples processed".format(len(answers), len(dataset)))
        return answers

    def train(self, optimizer,
              training_set: Iterable[Tuple[QASetting, List[Answer]]],
              max_epochs=10, hooks=[],
              l2=0.0, clip=None, clip_op=tf.clip_by_value,
              dataset_name=None):
        """
        This method trains the reader (and changes its state).
        
        Args:
            optimizer: TF optimizer
            training_set: the training instances.
            max_epochs: maximum number of epochs
            hooks: TrainingHook implementations that are called after epochs and batches
            l2: whether to use l2 regularization
            clip: whether to apply gradient clipping and at which value
            clip_op: operation to perform for clipping
        """
        assert self.is_train, "Reader has to be created for with is_train=True for training."
        logger.info("Setting up data and model...")
        # First setup shared resources, e.g., vocabulary. This depends on the input module.
        self.setup_from_data(training_set, dataset_name, "train")
        self.session.run([v.initializer for v in self.model_module.variables])

        batches = self.input_module.batch_generator(training_set, is_eval=False, dataset_name=dataset_name,
                                                    identifier='train')
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        loss = self.model_module.tensors[Ports.loss]

        if l2:
            loss += \
                tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2

        if clip:
            gradients = optimizer.compute_gradients(loss)
            if clip_op == tf.clip_by_value:
                gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                             for grad, var in gradients if grad]
            elif clip_op == tf.clip_by_norm:
                gradients = [(tf.clip_by_norm(grad, clip), var)
                             for grad, var in gradients if grad]
            min_op = optimizer.apply_gradients(gradients)
        else:
            min_op = optimizer.minimize(loss)

        # initialize non model variables like learning rate, optimizer vars ...
        self.session.run([v.initializer for v in tf.global_variables() if v not in self.model_module.variables])

        logger.info("Start training...")
        for i in range(1, max_epochs + 1):
            for j, batch in enumerate(batches):
                feed_dict = self.model_module.convert_to_feed_dict(batch)

                current_loss, _ = self.session.run([loss, min_op], feed_dict=feed_dict)

                for hook in hooks:
                    hook.at_iteration_end(i, current_loss, set_name='train')

            # calling post-epoch hooks
            for hook in hooks:
                hook.at_epoch_end(i)

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]], dataset_name=None, identifier='train'):
        """
        Sets up modules given a training dataset if necessary.
        
        Args:
            data: training dataset
        """
        self.input_module.setup_from_data(data, dataset_name, identifier)
        self.input_module.setup()
        self.model_module.setup(self.is_train)
        self.output_module.setup()
        self.session.run([v.initializer for v in self.model_module.variables])

    def load_and_setup(self, path):
        """
        Sets up already stored reader from model directory.
        
        Args:
            path: training dataset
        """
        self.shared_resources.load(os.path.join(path, "shared_resources"))
        self.input_module.setup()
        self.input_module.load(os.path.join(path, "input_module"))
        self.model_module.setup(self.is_train)
        self.session.run([v.initializer for v in self.model_module.variables])
        self.model_module.load(self.session, os.path.join(path, "model_module"))
        self.output_module.setup()
        self.output_module.load(os.path.join(path, "output_module"))

    def load(self, path):
        """
        (Re)loads module states on a setup reader (but not shared resources).
        If reader is not setup yet use setup from file instead.
        
        Args:
            path: model directory
        """
        self.input_module.load(os.path.join(path, "input_module"))
        self.model_module.load(self.session, os.path.join(path, "model_module"))
        self.output_module.load(os.path.join(path, "output_module"))

    def store(self, path):
        """
        Store module states and shared resources.
        
        Args:
            path: model directory
        """
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.shared_resources.store(os.path.join(path, "shared_resources"))
        self.input_module.store(os.path.join(path, "input_module"))
        self.model_module.store(self.session, os.path.join(path, "model_module"))
        self.output_module.store(os.path.join(path, "output_module"))
