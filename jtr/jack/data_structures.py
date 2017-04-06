# -*- coding: utf-8 -*-

"""
Here we define light data structures to store the input to jtr readers, and their output.
"""

import collections
import json
from typing import NamedTuple, List, Tuple


def NamedTupleWithDefaults(typename, fields, default_values=()):
    T = NamedTuple(typename, fields)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


Answer = NamedTuple("Answer", [('text', str), ('span', Tuple[int, int]), ('score', float)])
QASetting = NamedTuple("QASetting", [('question', str),
                                     ('support', List[str]),
                                     # id of the instance
                                     ('id', str),
                                     # candidates if any
                                     ('atomic_candidates', List[str]),
                                     ('seq_candidates', List[List[str]]),
                                     ('candidate_spans', List[Tuple[int, int]])])


# Wrapper for creating input
def QASettingWithDefaults(question, support=(), id=None,
                          atomic_candidates=None, seq_candidates=None, candidate_spans=None):
    return QASetting(question, support, id,
                     atomic_candidates, seq_candidates, candidate_spans)


def AnswerWithDefault(text: str, span: Tuple[int, int] = None, score: float = 1.0):
    return Answer(text, span, score)


def convert2qasettings(jtr_data, max_count=None, **options):
    def value(c, key="text"):
        if isinstance(c, dict):
            return c.get(key, None)
        elif key != "text":
            return None
        else:
            return c

    global_candidates = None
    if "globals" in jtr_data:
        global_candidates = [value(c) for c in jtr_data['globals']['candidates']]

    def convert_instance(instance):
        support = [value(s) for s in instance["support"]] if "support" in instance else None
        for question_instance in instance["questions"]:
            question = value(question_instance['question'])
            idd = value(question_instance['question'], 'id')
            if global_candidates is None:
                candidates = [value(c) for c in
                              question_instance['candidates']] if "candidates" in question_instance else None
            else:
                candidates = global_candidates
            answers = [Answer(value(c), value(c, 'span'), 1.0)
                       for c in question_instance['answers']] if "answers" in question_instance else None
            yield QASettingWithDefaults(question, support, atomic_candidates=candidates, id=idd), answers

    result = [(inp, answer) for i in jtr_data["instances"] for inp, answer in convert_instance(i)][:max_count]
    if max_count is None:
        return result
    else:
        return result[:max_count]


def load_labelled_data(path, max_count=None, **options) -> List[Tuple[QASetting, List[Answer]]]:
    """
    This function loads a jtr json file with labelled answers from a specific location.
    Args:
        path: the location to load from.
        max_count: how many instances to load at most
        **options: options to pass on to the loader. TODO: what are the options

    Returns:
        A list of input-answer pairs.

    """
    # We load json directly instead
    with open(path) as f:
        jtr_data = json.load(f)

    return convert2qasettings(jtr_data, max_count, **options)
