# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import Sequence, Mapping

import numpy as np

from jack.core.data_structures import QASetting, Answer
from jack.core.tensorport import TensorPort


class OutputModule:
    """
    An output module takes the output (numpy) tensors of the model module and turns them into
    jack data structures.
    """

    @property
    @abstractmethod
    def input_ports(self) -> Sequence[TensorPort]:
        """Returns: correspond to a subset of output ports of model module."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, questions: Sequence[QASetting], tensors: Mapping[TensorPort, np.array]) \
            -> Sequence[Answer]:
        """
        Process the tensors corresponding to the defined `input_ports` for a batch to produce a list of answers.
        The module has access to the original inputs.
        Args:
            questions:
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
