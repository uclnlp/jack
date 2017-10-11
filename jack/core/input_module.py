# -*- coding: utf-8 -*-

import logging
import random
from abc import abstractmethod
from typing import Iterable, Tuple, List, Mapping, TypeVar, Generic, Optional

import numpy as np

from jack.core.data_structures import QASetting, Answer
from jack.core.tensorport import TensorPort
from jack.util.batch import shuffle_and_batch, GeneratorWithRestart

_rng = random.Random(1234)
logger = logging.getLogger(__name__)


class InputModule:
    """An input module processes inputs and turns them into tensors to be processed by the model module.

    Note that all setting up should be done in the setup method, NOT in the constructor. Only use the constructor to
    hand over external variables/states, like `SharedResources`.
    """

    def setup(self):
        """Optionally, sets up the module (if needs setup after loading shared resources for instance).

        Assumes shared resources are fully setup. Usually called after loading and after `setup_from_data` as well."""
        pass

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        """Optionally, sets up the module based on input data.

        This usually involves setting up vocabularies and other resources. This
        should and is only called before training, not before loading a saved model.

        Args:
            data: a set of pairs of input and answer.
        """
        pass

    @property
    @abstractmethod
    def output_ports(self) -> List[TensorPort]:
        """
        Defines what types of tensors the output module produces in each batch.
        Returns: a list of tensor ports that correspond to the tensor ports in the mapping
        produced by `__call__`. The `batch_generator` method will return bindings for these
        ports and the ones in `training_ports`.
        """
        raise NotImplementedError

    @property
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
    def batch_generator(self, dataset: Iterable[Tuple[QASetting, List[Answer]]], batch_size: int,
                        is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
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
    """InputModule that preprocesses datasets on the fly.

    It provides implementations for `create_batch()` and `__call__()` and
    introduces two abstract methods:
    - `preprocess()`: Converts a list of instances to annotations.
    - `create_batch()`: Converts a list of annotations to a batch.

    Both of these methods are parameterized by `AnnotationType`. In the simplest
    case, this could be a `dict`, but you could also define a separate class
    for your annotation, in order to get stronger typing.
    """

    @abstractmethod
    def preprocess(self, questions: List[QASetting], answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) -> List[AnnotationType]:
        """Preprocesses a list of samples, returning a list of annotations.

        Batches of these annotation objects are then passed to the the `create_batch` method.

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
                     is_eval: bool, with_answers: bool) -> Mapping[TensorPort, np.ndarray]:
        """Creates a batch from a list of preprocessed questions.

        These are given by a list of annotations as returned by `preprocess_instance`.
        Args:
            annotations: a list of annotations to be included in the batch
            is_eval: whether the method is called for evaluation data
            with_answers: whether answers are included in the annotations

        Returns:
            A mapping from ports to numpy arrays.
        """

        raise NotImplementedError

    def batch_annotations(self, annotations: List[AnnotationType], batch_size, is_eval: bool):
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
        return shuffle_and_batch(annotations, batch_size, rng)

    def shuffle(self, is_eval: bool) -> bool:
        """Whether to shuffle the dataset in batch_annotations(). Default is noe is_eval."""
        return not is_eval

    def __call__(self, qa_settings: List[QASetting]) -> Mapping[TensorPort, np.ndarray]:
        """Preprocesses all qa_settings, returns a single batch with all instances."""

        annotations = self.preprocess(qa_settings, answers=None, is_eval=True)
        return self.create_batch(annotations, is_eval=True, with_answers=False)

    def batch_generator(self, dataset: Iterable[Tuple[QASetting, List[Answer]]], batch_size: int, is_eval: bool) \
            -> Iterable[Mapping[TensorPort, np.ndarray]]:
        """Preprocesses all instances, batches & shuffles them and generates batches in dicts."""
        questions, answers = zip(*dataset)
        annotations = self.preprocess(questions, answers, is_eval=is_eval)

        def make_generator():
            for annotation_batch in self.batch_annotations(annotations, batch_size, is_eval):
                yield self.create_batch(annotation_batch, is_eval, True)

        return GeneratorWithRestart(make_generator)
