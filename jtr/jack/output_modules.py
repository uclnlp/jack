from jtr.jack.core import *
from typing import List

class ClassificationOutputModule(OutputModule):


    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores,
                Ports.Prediction.candidate_idx,
                Ports.Targets.candidate_idx]

    def __call__(self, inputs: List[QASetting],
                       logits,
                        argmax_value,
                        labels)-> List[Answer]:

        return [logits, argmax_value, labels]

    def setup(self):
        pass

    def store(self, path):
        pass

    def load(self, path):
        pass


