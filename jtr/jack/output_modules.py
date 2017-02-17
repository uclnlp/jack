from jtr.jack.core import *
from typing import List

class ClassificationOutputModule(OutputModule):


    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_index, Ports.Input.atomic_candidates]

    @abstractmethod
    def __call__(self, inputs: List[QASetting], *tensor_inputs: np.ndarray) -> List[Answer]:
        pass

    @abstractmethod
    def setup(self):
        pass

    def store(self, path):
        pass

    def load(self, path):
        pass


