from quebap.tensorizer import *
from quebap.util.tfutil import *
import json

def create_global_answer_candidates(candidates):
    return {
        'candidates': [{'text': cand} for cand in candidates]
    }


def create_question_global_candidates(question, answer):
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


def test_sequence_tensorizer():
    tensorizer = SequenceTensorizer(data)
    batch = next(tensorizer.create_batches(data, 2))
    converted = tensorizer.convert_to_predictions(batch[tensorizer.candidates], batch[tensorizer.target_values])
    assert converted[0]['questions'][0]['answers'][0] == {'score': 1.0, 'text': 'A,B'}
    assert converted[1]['questions'][0]['answers'][0] == {'score': 1.0, 'text': 'C,A'}


def test_qa_tensorizer():
    with open('./quebap/data/LS/snippet_quebapformat.json') as data_file:
    #with open('../quebap/quebap/data/SNLI/snippet_quebapformat.json') as data_file:
        data = json.load(data_file)

    tensorizer = GenericTensorizer(data)
    feed_dict = next(tensorizer.create_batches(data, batch_size=4))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for placeholder in feed_dict:
            print(placeholder)
            print_tensor_shape_op = tf.Print(placeholder, [tf.shape(placeholder)], "shape: ")
            print(sess.run(print_tensor_shape_op, feed_dict=feed_dict))
            print()

#test_sequence_tensorizer()
test_qa_tensorizer()
