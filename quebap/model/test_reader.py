from quebap.model.reader import *


def create_global_answer_candidates(candidates):
    return {
        'candidates': [{'text': cand} for cand in candidates]
    }


def create_question_global_candidates(question: str, answer: str):
    return {
        'question': question,
        'candidates': "#/globals/candidates",
        'answers': [{'text': answer}]

    }


def create_reading_instance(question, support=()):
    return {
        'support': support,
        'questions': (question,)
    }


data = {
    'globals': create_global_answer_candidates(('A,B', 'B,C', 'C,A')),
    'instances': (
        create_reading_instance(
            create_question_global_candidates("path#appos|->appos->president->poss->|poss", 'A,B')),
        create_reading_instance(
            create_question_global_candidates("works_for", 'C,A'))
    )
}


def test_sequence_batcher():
    batcher = SequenceBatcher(data)
    batch = next(batcher.create_batches(data, 2))
    converted = batcher.convert_to_predictions(batch[batcher.candidates], batch[batcher.target_values])
    assert converted[0]['questions'][0]['answers'][0] == {'score': 1.0, 'text': 'A,B'}
    assert converted[1]['questions'][0]['answers'][0] == {'score': 1.0, 'text': 'C,A'}

test_sequence_batcher()
