from quebap.model.reader import *


def create_global_answer_candidates(candidates):
    return {
        'candidates': [{'text': cand} for cand in candidates]
    }


def create_question_global_candidates(question: str):
    return {
        'question': question,
        'candidates': "#/globals/candidates"
    }


def create_reading_instance(questions, support=()):
    return {
        'support': support,
        'questions': questions
    }


data = {
    'globals': create_global_answer_candidates(('A,B', 'B,C', 'C,A')),
    'instances': (
        create_reading_instance((create_question_global_candidates("path#appos|->appos->president->poss->|poss"),)),
        create_reading_instance((create_question_global_candidates("works_for"),))
    )
}


def test_sequence_batcher():
    batcher = SequenceBatcher(data)
    print(batcher.all_candidate_tokens)


    print(batcher.all_question_tokens)

test_sequence_batcher()