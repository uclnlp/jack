"""
Here we define light data structures to store the input to jtr readers, and their output.
"""

from typing import NamedTuple, List, Tuple
from jtr.load.read_jtr import jtr_load

import collections


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
Input = NamedTuple("QASetting", [('question', str),
                                 ('support', List[str]),
                                 # id of the instance
                                 ('id', str),
                                 # candidates if any
                                 ('atomic_candidates', List[str]),
                                 ('seq_candidates', List[List[str]]),
                                 ('candidate_spans', List[Tuple[int, int]])])


# Wrapper for creating input
def InputWithDefaults(question, support, id=None,
                      atomic_candidates=None, seq_candidates=None, candidate_spans=None):
    return Input(question, support, id,
                 atomic_candidates, seq_candidates, candidate_spans)

def AnswerWithDefault(text: str, span: Tuple[int, int]=None, score: float=1.0):
    return Answer(text, span, score)


def load_labelled_data(path, max_count=None, **options) -> List[Tuple[Input, List[Answer]]]:
    """
    This function loads a jtr json file with labelled answers from a specific location.
    Args:
        path: the location to load from.
        max_count: how many instances to load at most
        **options: options to pass on to the loader. TODO: what are the options

    Returns:
        A list of input-answer pairs.

    """
    dict_data = jtr_load(path, max_count, **options)
    if "support" not in dict_data:
        dict_data["support"] = []

    def to_list(text_or_list):
        if isinstance(text_or_list, str):
            return [text_or_list]
        else:
            return text_or_list

    def convert_instance(index):
        support = to_list(dict_data['support'][index])
        question = dict_data['question'][index]
        candidates = to_list(dict_data['candidates'][index])
        answer = to_list(dict_data['answer'][index])
        answer_spans = to_list(dict_data['answer_spans'][index])
        return InputWithDefaults(question, support, atomic_candidates=candidates), \
               [Answer(a, s, 1.0) for a, s in zip(answer, answer_spans)]

    result = [convert_instance(i) for i in range(0, len(dict_data['question']))]
    return result
