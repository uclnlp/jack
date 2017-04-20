# -*- coding: utf-8 -*-

from jtr.jack.core import InputModule, Ports, TensorPort, Iterable, SharedResources
from jtr.jack.data_structures import QASetting, Answer

from jtr.preprocess.batch import get_batches
from jtr.jack.preprocessing import preprocess_with_pipeline

from typing import List, Tuple, Mapping
import numpy as np


class SingleSupportFixedClassInputs(InputModule):
    def __init__(self, shared_vocab_config):
        self.shared_vocab_config = shared_vocab_config

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Input.candidates1d]

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
                Ports.Input.question, Ports.Input.support_length,
                Ports.Input.question_length, Ports.Targets.candidate_idx, Ports.Input.sample_id]

    def __call__(self, qa_settings: List[QASetting]) -> Mapping[TensorPort, np.ndarray]:
        pass

    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        corpus, train_vocab, train_answer_vocab, train_candidate_vocab = \
                preprocess_with_pipeline(data, self.shared_vocab_config.vocab,
                        None, sepvocab=True)
        train_vocab.freeze()
        train_answer_vocab.freeze()
        train_candidate_vocab.freeze()
        self.shared_vocab_config.config['answer_size'] = len(train_answer_vocab)
        self.shared_vocab_config.vocab = train_vocab
        self.answer_vocab = train_answer_vocab

    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus, _, _, _ = preprocess_with_pipeline(dataset, self.shared_vocab_config.vocab, self.answer_vocab,
                                                   use_single_support=True, sepvocab=True)

        xy_dict = {
            Ports.Input.single_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Targets.candidate_idx:  corpus["answers"],
            Ports.Input.question_length : corpus['question_lengths'],
            Ports.Input.support_length : corpus['support_lengths'],
            Ports.Input.sample_id : corpus['ids']
        }

        return get_batches(xy_dict)

    def setup(self):
        pass
