from quebap.model.tensorizer import *
from quebap.util.tfutil import *

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


def test_sequence_tensorizer():
    tensorizer = SequenceTensorizer(data)
    batch = next(tensorizer.create_batches(data, 2))
    converted = tensorizer.convert_to_predictions(batch[tensorizer.candidates], batch[tensorizer.target_values])
    assert converted[0]['questions'][0]['answers'][0] == {'score': 1.0, 'text': 'A,B'}
    assert converted[1]['questions'][0]['answers'][0] == {'score': 1.0, 'text': 'C,A'}


qa_data = \
    {
      "meta": "SNLI",
      "globals": {
        "candidates": [
          {
            "text": "entailment"
          },
          {
            "text": "neutral"
          },
          {
            "text": "contradiction"
          }
        ]
      },
      "instances": [
        {
          "support": [
            {
              "text": "A boy is jumping on skateboard in the middle of a red bridge.",
              "id": "3691670743.jpg#0"
            }
          ],
          "questions": [
            {
              "answers": [
                {
                  "text": "neutral"
                }
              ],
              "question": "The boy is wearing safety equipment."
            }
          ],
          "id": "3691670743.jpg#0r1n"
        },
        {
          "support": [
            {
              "text": "An older man sits with his orange juice at a small table in a coffee shop while employees in bright colored shirts smile in the background.",
              "id": "4804607632.jpg#0"
            }
          ],
          "questions": [
            {
              "answers": [
                {
                  "text": "neutral"
                }
              ],
              "question": "An older man drinks his juice as he waits for his daughter to get off work."
            }
          ],
          "id": "4804607632.jpg#0r1n"
        }
      ]
    }


def test_qa_tensorizer():
    tensorizer = QATensorizer(qa_data)
    feed_dict = next(tensorizer.create_batches(qa_data, batch_size=2))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for placeholder in feed_dict:
            print(placeholder)
            print_tensor_shape_op = tf.Print(placeholder, [tf.shape(placeholder)], "shape: ")
            print(sess.run(print_tensor_shape_op, feed_dict=feed_dict))
            print()

#test_sequence_tensorizer()
test_qa_tensorizer()
