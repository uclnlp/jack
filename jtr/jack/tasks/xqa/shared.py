"""
This file contains reusable modules for extractive QA models and ports
"""

from jtr.jack.core import *
from jtr.preprocess.map import deep_map

class ParameterTensorPorts:
        # remove?
        keep_prob = TensorPortWithDefault(1.0, tf.float32, [], "keep_prob",
                                          "scalar representing keep probability when using dropout",
                                          "[]")

        is_eval = TensorPortWithDefault(True, tf.bool, [], "is_eval",
                                        "boolean that determines whether input is eval or training.",
                                        "[]")

class XQAPorts:
    # When feeding embeddings directly
    emb_question = FlatPorts.Misc.embedded_question
    question_length = FlatPorts.Input.question_length
    emb_support = FlatPorts.Misc.embedded_support
    support_length = FlatPorts.Input.support_length

    # but also ids, for char-based embeddings
    unique_word_chars = TensorPort(tf.int32, [None, None], "question_chars",
                                   "Represents questions using symbol vectors",
                                   "[U, max_num_chars]")
    unique_word_char_length = TensorPort(tf.int32, [None], "question_char_length",
                                         "Represents questions using symbol vectors",
                                         "[U]")
    question_words2unique = TensorPort(tf.int32, [None, None], "question_words2unique",
                                       "Represents support using symbol vectors",
                                       "[batch_size, max_num_question_tokens]")
    support_words2unique = TensorPort(tf.int32, [None, None], "support_words2unique",
                                      "Represents support using symbol vectors",
                                      "[batch_size, max_num_support_tokens, max]")

    keep_prob = ParameterTensorPorts.keep_prob
    is_eval = ParameterTensorPorts.is_eval

    # This feature is model specific and thus, not part of the conventional Ports
    word_in_question = TensorPort(tf.float32, [None, None], "word_in_question_feature",
                                  "Represents a 1/0 feature for all context tokens denoting"
                                  " whether it is part of the question or not",
                                  "[Q, support_length]")

    correct_start_training = TensorPortWithDefault(np.array([0], np.int32), tf.int32, [None], "correct_start_training",
                                                   "Represents the correct start of the span which is given to the"
                                                   "model during training for use to predicting end.",
                                                   "[A]")

    answer2question_training = TensorPortWithDefault([0], tf.int32, [None], "answer2question_training",
                                                     "Represents mapping to question idx per answer, which is used "
                                                     "together with correct_start_training during training.",
                                                     "[A]")

    # output ports
    start_scores = FlatPorts.Prediction.start_scores
    end_scores = FlatPorts.Prediction.end_scores
    span_prediction = FlatPorts.Prediction.answer_span
    token_char_offsets = TensorPort(tf.int32, [None, None], "token_char_offsets",
                                    "Character offsets of tokens in support.",
                                    "[S, support_length]")

    # ports used during training
    answer2question = FlatPorts.Input.answer2question
    answer_span = FlatPorts.Target.answer_span




def _np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class XQAOutputModule(OutputModule):
    def __init__(self, shared_vocab_confg: SharedVocabAndConfig):
        self.vocab = shared_vocab_confg.vocab
        self.setup()

    def __call__(self, questions, span_prediction, token_char_offsets, start_scores, end_scores) -> List[Answer]:
        answers = []
        for i, q in enumerate(questions):
            start, end = span_prediction[i, 0], span_prediction[i, 1]
            char_start = token_char_offsets[i, start]
            if end + 1 < token_char_offsets.shape[1]:
                char_end = token_char_offsets[i, end + 1]
                if char_end == 0:
                    char_end = len(q.support[0])
            else:
                char_end = len(q.support[0])
            answer = q.support[0][char_start: char_end]

            start_probs = _np_softmax(start_scores[i])
            end_probs = _np_softmax(end_scores[i])

            answer = answer.rstrip()
            char_end = char_start + len(answer)

            answers.append(Answer(answer, (char_start, char_end), score=start_probs[start] * end_probs[end]))

        return answers

    @property
    def input_ports(self) -> List[TensorPort]:
        return [FlatPorts.Prediction.answer_span, XQAPorts.token_char_offsets,
                FlatPorts.Prediction.start_scores, FlatPorts.Prediction.end_scores]


class XQANoScoreOutputModule(OutputModule):
    def __init__(self, shared_vocab_confg: SharedVocabAndConfig):
        self.vocab = shared_vocab_confg.vocab
        self.setup()

    def __call__(self, questions, span_prediction, token_char_offsets) -> List[Answer]:
        answers = []
        for i, q in enumerate(questions):
            start, end = span_prediction[i, 0], span_prediction[i, 1]
            char_start = token_char_offsets[i, start]
            if end + 1 < token_char_offsets.shape[1]:
                char_end = token_char_offsets[i, end + 1]
                if char_end == 0:
                    char_end = len(q.support[0])
            else:
                char_end = len(q.support[0])
            answer = q.support[0][char_start: char_end]

            answer = answer.rstrip()
            char_end = char_start + len(answer)

            answers.append(Answer(answer, (char_start, char_end), score=1.0))

        return answers

    @property
    def input_ports(self) -> List[TensorPort]:
        return [FlatPorts.Prediction.answer_span, XQAPorts.token_char_offsets]
