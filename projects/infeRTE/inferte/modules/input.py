# -*- coding: utf-8 -*-

from jtr.jack.core import InputModule, Ports, TensorPort, Iterable
from jtr.preprocess.batch import get_batches

from typing import List, Mapping, Dict, Any
import numpy as np


class SingleSupportFixedClassInputs(InputModule):
    def __init__(self):
        pass

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Target.target_index]

    @property
    def output_ports(self) -> List[TensorPort]:
        """Defines the outputs of the InputModule

        1. Word embedding index tensor of questions of mini-batchs
        2. Word embedding index tensor of support of mini-batchs
        3. Max timestep length of mini-batches for support tensor
        4. Max timestep length of mini-batches for question tensor
        5. Labels
        """
        return [Ports.Input.multiple_support,
                Ports.Input.question, Ports.Input.support_length,
                Ports.Input.question_length, Ports.Target.target_index, Ports.Input.sample_id]

    def __call__(self, qa_settings: List[Mapping]) -> Mapping[TensorPort, np.ndarray]:
        pass

    def dataset_generator(self, corpus: Dict[str, Any]) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        xy_dict = {
            Ports.Input.multiple_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Target.target_index:  corpus["answers"],
            Ports.Input.question_length: corpus['question_lengths'],
            Ports.Input.support_length: corpus['support_lengths'],
            Ports.Input.sample_id: corpus['ids']
        }
        return get_batches(xy_dict)

    def setup(self):
        pass
