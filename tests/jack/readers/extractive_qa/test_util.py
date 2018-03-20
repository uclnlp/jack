# -*- coding: utf-8 -*-

from jack.core import QASetting, Answer
from jack.readers.extractive_qa.util import prepare_data
from jack.util.vocab import Vocab

qa_setting = QASetting(question="What is the answer?",
                       support=["It is not A.", "It is B."])
answers = [Answer(text="B", span=(6, 7), doc_idx=1)]


def test_prepare_data():

    result = prepare_data(qa_setting, answers, Vocab(),
                          with_answers=True)

    question_tokens, question_ids, question_lemmas, question_length, \
    support_tokens, support_ids, support_lemmas, support_length, \
    word_in_question, token_offsets, answer_spans = result

    assert question_tokens == ['What', 'is', 'the', 'answer', '?']
    assert question_ids == [1, 2, 3, 4, 5]
    assert question_lemmas is None
    assert question_length == 5

    assert support_tokens == [['It', 'is', 'not', 'A', '.', ], ['It', 'is', 'B', '.']]
    assert support_ids == [[6, 2, 7, 8, 9], [6, 2, 10, 9]]
    assert support_lemmas == [None, None]
    assert support_length == [5, 4]
    assert word_in_question == [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    assert token_offsets == [[0, 3, 6, 10, 11], [0, 3, 6, 7]]
    assert answer_spans == [[], [(2, 2)]]
