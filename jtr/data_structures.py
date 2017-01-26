"""
Here we define light data structures to store the input to jtr readers, and their output.
"""

from typing import NamedTuple, List, Tuple

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
