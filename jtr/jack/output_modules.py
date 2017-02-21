from jtr.jack.core import *
from typing import List

class ClassificationOutputModule(OutputModule):


    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_index,
                Ports.Targets.target_index]

    def __call__(self, inputs: List[QASetting],
                        candiate_index,
                        atomic_candidates)-> List[Answer]:

        return [candidate_index, atomic_candidates]

    def setup(self):
        pass

    def store(self, path):
        pass

    def load(self, path):
        pass


