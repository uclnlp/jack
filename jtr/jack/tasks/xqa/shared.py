"""
This file contains reusable modules for extractive QA models and ports
"""

from jtr.jack import *


class XqaPorts:
    token_char_offsets = TensorPort(tf.int32, [None, None], "token_char_offsets",
                                    "Character offsets of tokens in support.",
                                    "[S, support_length]")


def _np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class XqaOutputModule(OutputModule):
    def __call__(self, questions, span_prediction, token_char_offsets, start_scores, end_scores) -> List[Answer]:
        answers = []
        for i, q in enumerate(questions):
            start, end = span_prediction[i, 0], span_prediction[i, 1]
            char_start = token_char_offsets[i, start]
            char_end = token_char_offsets[i, end+1]
            answer = q.support[0][char_start: char_end]

            start_probs = _np_softmax(start_scores[i])
            end_probs = _np_softmax(end_scores[i])

            #strip answer
            while answer[-1].isspace():
                answer = answer[:-1]
                char_end -= 1

            answers.append(AnswerWithDefault(answer, (char_start, char_end), score=start_probs[start] * end_probs[end]))

        return answers

    @property
    def input_ports(self) -> List[TensorPort]:
        return [FlatPorts.Prediction.answer_span, XqaPorts.token_char_offsets,
                FlatPorts.Prediction.start_scores, FlatPorts.Prediction.end_scores]

    def setup(self, shared_resources):
        self.vocab = shared_resources.vocab
