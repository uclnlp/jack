# -*- coding: utf-8 -*-
import logging
from abc import abstractmethod
from typing import Mapping, List, Sequence

import numpy as np

from jack.core.tensorport import TensorPort

logger = logging.getLogger(__name__)


class ModelModule:
    """A model module defines the actual reader model by processing input tensors and producing output tensors.

    A model module encapsulates two computations (possibly overlapping): one which computes all
    predictions (to be processed by the output module) and another representing the loss(es) and potenially other
    training related outputs. It defines the expected input and output tensor shapes and types via its respective input
    and output pairs.
    """

    @abstractmethod
    def __call__(self, batch: Mapping[TensorPort, np.ndarray],
                 goal_ports: List[TensorPort] = None) -> Mapping[TensorPort, np.ndarray]:
        """Runs a batch and returns values/outputs for specified goal ports.
        Args:
            batch: mapping from ports to values
            goal_ports: optional output ports, defaults to output_ports of this module will be returned

        Returns:
            A mapping from goal ports to tensors.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_ports(self) -> Sequence[TensorPort]:
        """Returns: Definition of the output ports of this module (predictions made by this model)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def input_ports(self) -> Sequence[TensorPort]:
        """Returns: Definition of the input ports."""
        raise NotImplementedError

    @property
    @abstractmethod
    def training_input_ports(self) -> Sequence[TensorPort]:
        """Returns: Definition of the input ports necessary to create the training output ports, i.e., they do not have
        to be provided during eval and they can include output ports of this module."""
        raise NotImplementedError

    @property
    @abstractmethod
    def training_output_ports(self) -> Sequence[TensorPort]:
        """Returns: Definition of the output ports provided during training for this module (usually just the loss)."""
        raise NotImplementedError

    @abstractmethod
    def setup(self, is_training=True, reuse=False):
        """Sets up the module."""
        raise NotImplementedError

    @abstractmethod
    def store(self, path):
        """Store the state of this module."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        """Load the state of this module."""
        raise NotImplementedError
