# -*- coding: utf-8 -*-

from jtr.jack.core import InputModule, Ports, TensorPort, Iterable, SharedResources
from jtr.jack.data_structures import QASetting, Answer

from jtr.preprocess.batch import get_batches

from jtr.preprocess.map import tokenize, notokenize, lower, deep_map, deep_seq_map, dynamic_subsample
from jtr.preprocess.vocab import Vocab

from typing import List, Tuple, Mapping
import numpy as np


def pipeline(corpus, vocab=None, target_vocab=None, candidate_vocab=None,
             emb=None, freeze=False, normalize=False, tokenization=True, lowercase=True,
             negsamples=0, sepvocab=True, test_time=False, cache_fun=False, map_to_target=True):
    vocab = vocab or Vocab(emb=emb)
    if sepvocab:
        target_vocab = target_vocab or Vocab(unk=None)
        candidate_vocab = candidate_vocab or Vocab(unk=None)
    if freeze:
        vocab.freeze()
        if sepvocab:
            target_vocab.freeze()
            candidate_vocab.freeze()

    if not sepvocab:
        target_vocab = candidate_vocab = vocab

    corpus_tokenized = deep_map(corpus, tokenize if tokenization else notokenize, ['question', 'support'])
    corpus_lower = deep_seq_map(corpus_tokenized, lower, ['question', 'support']) if lowercase else corpus_tokenized
    corpus_os = deep_seq_map(corpus_lower, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ['question', 'support'])\
        if tokenization else corpus_lower

    corpus_ids = deep_map(corpus_os, vocab, ['question', 'support'])
    if not test_time:
        corpus_ids = deep_map(corpus_ids, target_vocab, ['answers'])
    corpus_ids = deep_map(corpus_ids, candidate_vocab, ['candidates'], cache_fun=cache_fun)
    if map_to_target and not test_time:
        def jtr_map_to_targets(xs, cands_name, ans_name):
            """
            Create cand-length vector for each training instance with 1.0s for cands which are the correct answ and 0.0s for cands which are the wrong answ
            #@todo: integrate this function with the one below - the pipeline() method only works with this function
            """
            xs["targets"] = [1.0 if xs[ans_name][i] == cand else 0.0
                             for i in range(len(xs[ans_name]))
                             for cand in xs[cands_name][i]]
            return xs
        corpus_ids = jtr_map_to_targets(corpus_ids, 'candidates', 'answers')
    corpus_ids = deep_seq_map(corpus_ids, lambda xs: len(xs), keys=['question', 'support'], fun_name='lengths', expand=True)
    if negsamples > 0 and not test_time:#we want this to be the last thing we do to candidates
            corpus_ids = dynamic_subsample(corpus_ids,'candidates','answers',how_many=negsamples)
    if normalize:
        corpus_ids = deep_map(corpus_ids, vocab._normalize, keys=['question', 'support'])
    return corpus_ids, vocab, target_vocab, candidate_vocab


def preprocess_with_pipeline(data, vocab, target_vocab, test_time=False, negsamples=0,
                             tokenization=True, use_single_support=True, sepvocab=True):
    corpus = {"support": [], "question": [], "candidates": [], "ids": []}
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

    corpus, train_vocab, answer_vocab, train_candidates_vocab =\
        pipeline(corpus, vocab, target_vocab, sepvocab=sepvocab, test_time=test_time,
                 tokenization=tokenization, cache_fun=True, map_to_target=False, normalize=True,
                 **({'negsamples': negsamples} if not test_time else {}))
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
