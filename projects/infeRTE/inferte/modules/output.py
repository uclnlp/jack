# -*- coding: utf-8 -*-

from jtr.jack.core import Ports, TensorPort, OutputModule
from jtr.jack.data_structures import QASetting, Answer

from typing import List
import numpy as np


class EmptyOutputModule(OutputModule):

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits,
                Ports.Prediction.candidate_index,
                Ports.Target.target_index]

    def __call__(self, inputs: List[QASetting], *tensor_inputs: np.ndarray) -> List[Answer]:
        return tensor_inputs

    def setup(self):
        pass

    def store(self, path):
        pass

    def load(self, path):
        pass