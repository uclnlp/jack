# -*- coding: utf-8 -*-

"""
Here we define light data structures to store the input to jtr readers, and their output.
"""

import json
import copy
from typing import List, Tuple, Sequence
from jtr.util.batch import GeneratorWithRestart


class Answer:
    """
    Representation of an answer to a question.
    """

    def __init__(self, text: str, span: Tuple[int, int, int]=None, score: float=1.0):
        """
        Create a new answer.
        Args:
            text: The text string of the answer.
            span: For extractive QA, a span in the support documents. The triple `(doc_index, start, end)`
            represents a span in support document with index `doc_index` in the ordered sequence of
            support documents. The span starts at `start` and ends at `end` (exclusive).
            score: the score a model associates with this answer.
        """
        self.score = score
        self.span = span
        self.text = text


class QASetting:
    """
    Representation of a single question answering problem. It primarily consists of a question,
    a list of support documents, and optionally, some set of candidate answers.
    """

    def __init__(self,
                 question: str,
                 support: Sequence[str] = (),
                 id: str = None,
                 atomic_candidates: Sequence[str] = None,
                 seq_candidates: Sequence[str] = None,
                 candidate_spans: Sequence[Tuple[int, int, int]]=None):
        """
        Create a new QASetting.
        Args:
            question: the question text.
            support: a sequence of support documents the answerer has access to when answering the question.
            id: an identifier for the problem.
            atomic_candidates: a list of candidate answer strings.
            seq_candidates: REMOVEME
            candidate_spans: for extractive QA, a sequence of candidate spans in the support documents.
            A span `(doc_index,start,end)` corresponds to a span in support document with index `doc_index`,
            with start position `start` and end position `end`.
        """
        self.id = id
        self.candidate_spans = candidate_spans
        self.seq_candidates = seq_candidates
        self.atomic_candidates = atomic_candidates
        self.support = support
        self.question = question


def convert2qasettings(jtr_data, max_count=None):
    """
    Converts a python dictionary to a QASetting.
    Args:
        jtr_data: dictionary extracted from jack jason file.
        max_count: maximal number of instances to load.

    Returns:

    """

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
            yield QASetting(question, support, atomic_candidates=candidates, id=idd), answers

    result = [(inp, answer) for i in jtr_data["instances"] for inp, answer in convert_instance(i)][:max_count]
    if max_count is None:
        return result
    else:
        return result[:max_count]


def load_labelled_data_stream(path, dataset_streamer):
        stream_processor = copy.deepcopy(dataset_streamer)

        stream_processor.set_path(path)

        data_set = GeneratorWithRestart(stream_processor.stream)
        return data_set


def load_labelled_data(path, max_count=None) -> List[Tuple[QASetting, List[Answer]]]:
    """
    This function loads a jtr json file with labelled answers from a specific location.
    Args:
        path: the location to load from.
        max_count: how many instances to load at most

    Returns:
        A list of input-answer pairs.

    """
    # We load json directly instead
    data = []
    if isinstance(path, list):
        for p in path:
            with open(p) as f:
                jtr_data = json.load(f)
            data += convert2qasettings(jtr_data, max_count)
        return data
    else:
        with open(path) as f:
            jtr_data = json.load(f)
        return convert2qasettings(jtr_data, max_count)
