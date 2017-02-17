from jtr.pipelines import pipeline
from jtr.preprocess.batch import get_batches
from jtr.preprocess.map import numpify
from jtr.preprocess.vocab import Vocab
from jtr.jack.preprocessing import preprocess_with_pipeline
from jtr.jack.core import *
from typing import List, Dict, Mapping, Tuple
import numpy as np

class QuestionOneSupportGlobalCandiatesInputModule(InputModule):
    def __init__(self, shared_vocab_config):
        self.shared_vocab_config = shared_vocab_config


    @property
    def output_ports(self) -> List[TensorPort]:
        """Defines the outputs of the InputModule

        1. Word embedding index tensor of questions of mini-batchs
        2. Word embedding index tensor of support of mini-batchs
        3. Max timestep length of mini-batches for support tensor
        4. Max timestep length of mini-batches for question tensor
        5. Labels
        """
        return [Ports.Input.single_support,  Ports.Input.single_support,
                Ports.Input.question, Ports.Input.support_length,
                Ports.Input.question_length]


    def __call__(self, qa_settings : List[QASetting]) \
                    -> Mapping[TensorPort, np.ndarray]:
        corpus = preprocess_with_pipeline(data, test_time)

        x_dict = {
            Ports.Input.single_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Input.question_length : corpus['question_lengths'],
            Ports.Input.support_length : corpus['support_lengths'],
            Ports.Input.atomic_candiates : corpus['candidates']
        }

        return numpify(x_dict)


    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        preprocess_with_pipeline(data, test_time)
        len(corpus['candidates'])
        pass


    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus = preprocess_with_pipeline(data, test_time, negsamples=1)
        xy_dict = {
            Ports.Input.single_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Input.atomic_candidates :  corpus["candidates"],
            Ports.Input.question_length : corpus['question_lengths'],
            Ports.Input.support_length : corpus['support_lengths']
        }
        if not is_eval:
            xy_dict[Ports.Targets.target_index] =  corpus["answers"]

        return get_batches(xy_dict)

    @abstractmethod
    def setup(self):
        pass
