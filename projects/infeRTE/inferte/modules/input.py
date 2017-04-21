# -*- coding: utf-8 -*-

from jtr.jack.core import InputModule, Ports, TensorPort, Iterable, SharedResources
from jtr.jack.data_structures import QASetting, Answer

from jtr.preprocess.batch import get_batches
from jtr.pipelines import pipeline

from typing import List, Tuple, Mapping
import numpy as np


def preprocess_with_pipeline(data, vocab, target_vocab, test_time=False, negsamples=0,
                             tokenization=True, use_single_support=True, sepvocab=True):
    corpus = {"support": [], "question": [], "candidates": [], "ids" : []}
    if not test_time:
        corpus["answers"] = []
    for i, xy in enumerate(data):
        x, y = (xy, None) if test_time else xy

        corpus["support"] += [x.support[0] if use_single_support else x.support]
        corpus['ids'].append(i)
        corpus["question"].append(x.question)
        corpus["candidates"].append(x.atomic_candidates)
        assert len(y) == 1
        if not test_time:
            corpus["answers"].append(y[0].text)
    if not test_time:
        corpus, train_vocab, answer_vocab, train_candidates_vocab =\
            pipeline(corpus, vocab, target_vocab, sepvocab=sepvocab, test_time=test_time,
                     tokenization=tokenization, negsamples=negsamples, cache_fun=True,
                     map_to_target=False, normalize=True)
    else:
        corpus, train_vocab, answer_vocab, train_candidates_vocab = \
            pipeline(corpus, vocab, target_vocab, sepvocab=sepvocab, test_time=test_time,
                     tokenization=tokenization, cache_fun=True, map_to_target=False, normalize=True)
    return corpus, train_vocab, answer_vocab, train_candidates_vocab


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
