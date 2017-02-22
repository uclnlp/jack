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
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Input.atomic_candidates]


    @property
    def output_ports(self) -> List[TensorPort]:
        """Defines the outputs of the InputModule

        1. Word embedding index tensor of questions of mini-batchs
        2. Word embedding index tensor of support of mini-batchs
        3. Max timestep length of mini-batches for support tensor
        4. Max timestep length of mini-batches for question tensor
        5. Labels
        """
        return [Ports.Input.single_support,
                Ports.Input.question, FlatPorts.Input.support_length,
                FlatPorts.Input.question_length, FlatPorts.Target.candidate_idx]


    def __call__(self, qa_settings : List[QASetting]) \
                    -> Mapping[TensorPort, np.ndarray]:
        pass
        #corpus, train_vocab, train_answer_vocab, train_candidate_vocab = \
        #        preprocess_with_pipeline(data, self.shared_vocab_config.vocab)

        #x_dict = {
        #    Ports.Input.single_support: corpus["support"],
        #    Ports.Input.question: corpus["question"],
        #    FlatPorts.Input.question_length : corpus['question_lengths'],
        #    FlatPorts.Input.support_length : corpus['support_lengths'],
        #    Ports.Input.atomic_candiates : corpus['candidates']
        #}

        #return numpify(x_dict)


    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        pass
        #corpus, train_vocab, train_answer_vocab, train_candidate_vocab = \
        #        preprocess_with_pipeline(data, self.shared_vocab_config.vocab)
        #print(len(train_candidate_vocab))
        #print(len(train_answer_vocab))
        #self.shared_vocab_config.config['answer_size'] = len(train_answer_vocab)


    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus, train_vocab, train_answer_vocab, train_candidate_vocab = \
                preprocess_with_pipeline(dataset, self.shared_vocab_config.vocab, use_single_support=True)
        xy_dict = {
            Ports.Input.single_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            FlatPorts.Target.candidate_idx:  corpus["answers"],
            FlatPorts.Input.question_length : corpus['question_lengths'],
            FlatPorts.Input.support_length : corpus['support_lengths']
        }
        if not is_eval:
            xy_dict[Ports.Targets.target_index] =  corpus["answers"]

        print(corpus['question_lengths'][0])

        keys = [t.name for t in xy_dict]
        return get_batches(xy_dict)

    @abstractmethod
    def setup(self):
        pass
