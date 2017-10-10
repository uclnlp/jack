# -*- coding: utf-8 -*-

"""
Here we define light data structures to store the input to jack readers, and their output.
"""

from typing import Tuple, Sequence


class Answer:
    """
    Representation of an answer to a question.
    """

    def __init__(self, text: str, span: Tuple[int, int] = None, doc_idx: int = None, score: float = 1.0):
        """
        Create a new answer.
        Args:
            text: The text string of the answer.
            span: For extractive QA, a span in the support documents. The triple `(start, end)`
                represents a span in support document with index `doc_index` in the ordered sequence of
            doc_idx: index of document where answer was found
            support documents. The span starts at `start` and ends at `end` (exclusive).
            score: the score a model associates with this answer.
        """
        self.score = score
        self.span = span
        self.doc_idx = doc_idx
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
                 candidate_spans: Sequence[Tuple[int, int, int]] = None):
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


def _jtr_to_qasetting(instance, value, global_candidates):
    support = [value(s) for s in instance["support"]] if "support" in instance else None
    for question_instance in instance["questions"]:
        question = value(question_instance['question'])
        idd = value(question_instance['question'], 'id')
        if global_candidates is None:
            candidates = [value(c) for c in question_instance['candidates']] if "candidates" in question_instance else None
        else:
            candidates = global_candidates
        answers = [Answer(value(c), value(c, 'span'), score=1.0) for c in question_instance['answers']] if "answers" in question_instance else None
        yield QASetting(question, support, atomic_candidates=candidates, id=idd), answers


def jtr_to_qasetting(jtr_data, max_count=None):
    """
    Converts a python dictionary in JTR format to a QASetting.
    Args:
        jtr_data: dictionary extracted from jack json file.
        max_count: maximal number of instances to load.

    Returns:
        list of QASetting
    """

    def value(c, key="text"):
        return c.get(key, None) if isinstance(c, dict) else c if key == 'text' else None

    global_candidates = [value(c) for c in jtr_data['globals']['candidates']] if 'globals' in jtr_data else None

    ans = [(inp, answer) for i in jtr_data["instances"]
           for inp, answer in _jtr_to_qasetting(i, value, global_candidates)][:max_count]
    return ans[:max_count]
