"""Shared utilities for multiple choice."""
from typing import Iterable

from jack.core.data_structures import QASetting, Answer
from jack.util.vocab import Vocab


def create_answer_vocab(qa_settings: Iterable[QASetting] = None, answers: Iterable[Answer] = None):
    vocab = Vocab(unk=None)
    if qa_settings is not None:
        for qa in qa_settings:
            if qa.candidates:
                for c in qa.candidates:
                    vocab(c)
    if answers is not None:
        for a in answers:
            vocab(a.text)
    return vocab


def candidate_one_hot(candidates, answer_str):
    return [1.0 if candidates[answer_str] == cand else 0.0 for cand in candidates]
