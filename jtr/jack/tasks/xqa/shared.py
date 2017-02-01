"""
This file contains reusable modules for extractive QA models and ports
"""

from jtr.jack import *


class XqaPorts:
    token_char_offsets = TensorPort(tf.int32, [None, None], "token_char_offsets",
                                    "Character offsets of tokens in support.",
                                    "[S, support_length]")


class XqaOutputModule(OutputModule):
    def __call__(self, inputs: List[Question], span_prediction:np.array, token_char_offsets:np.array) -> List[Answer]:
        answers = []
        for i, q in enumerate(inputs):
            start, end = span_prediction[i, 0], span_prediction[i, 1]
            char_start = token_char_offsets[start]
            char_end = token_char_offsets[end]
            answer = q.support[0][char_start: char_end]
            #strip answer
            while answer[-1].isspace():
                answer = answer[:-1]
                char_end -= 1

            answers.append(AnswerWithDefault(answer, (char_start, char_end)))

        return answers

    @property
    def input_ports(self) -> List[TensorPort]:
        return [FlatPorts.Prediction.answer_span, XqaPorts.token_char_offsets]

    def setup(self, shared_resources):
        self.vocab = shared_resources.vocab
