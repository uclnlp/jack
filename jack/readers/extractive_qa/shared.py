"""
This file contains reusable modules for extractive QA models and ports
"""
from typing import NamedTuple

from jack.core import *
from jack.readers.extractive_qa.util import unique_words_with_chars, prepare_data
from jack.tf_util.xqa import xqa_min_crossentropy_loss
from jack.util import preprocessing
from jack.util.map import numpify
from jack.util.preprocessing import stack_and_pad


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

    # output ports
    start_scores = FlatPorts.Prediction.start_scores
    end_scores = FlatPorts.Prediction.end_scores
    span_prediction = FlatPorts.Prediction.answer_span
    token_char_offsets = TensorPort(tf.int32, [None, None], "token_char_offsets",
                                    "Character offsets of tokens in support.",
                                    "[S, support_length]")

    # ports used during training
    answer2question_training = FlatPorts.Input.answer2question
    answer_span = FlatPorts.Target.answer_span


XQAAnnotation = NamedTuple('XQAAnnotation', [
    ('question_tokens', List[str]),
    ('question_ids', List[int]),
    ('question_length', int),
    ('question_embeddings', np.ndarray),
    ('support_tokens', List[str]),
    ('support_ids', List[int]),
    ('support_length', int),
    ('support_embeddings', np.ndarray),
    ('word_in_question', List[float]),
    ('token_offsets', List[int]),
    ('answer_spans', Optional[List[Tuple[int, int]]]),
])


class XQAInputModule(OnlineInputModule[XQAAnnotation]):
    _output_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                     XQAPorts.emb_support, XQAPorts.support_length,
                     # char
                     XQAPorts.unique_word_chars, XQAPorts.unique_word_char_length,
                     XQAPorts.question_words2unique, XQAPorts.support_words2unique,
                     # features
                     XQAPorts.word_in_question,
                     # optional, only during training
                     XQAPorts.correct_start_training, XQAPorts.answer2question_training,
                     XQAPorts.keep_prob, XQAPorts.is_eval,
                     # for output module
                     XQAPorts.token_char_offsets]
    _training_ports = [XQAPorts.answer_span, XQAPorts.answer2question_training]

    def __init__(self, shared_vocab_config):
        assert isinstance(shared_vocab_config, SharedResources), \
            "shared_resources for FastQAInputModule must be an instance of SharedResources"
        self.shared_vocab_config = shared_vocab_config

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        # create character vocab + word lengths + char ids per word
        self.shared_vocab_config.char_vocab = preprocessing.char_vocab_from_vocab(self.shared_vocab_config.vocab)

    def setup(self):
        self.vocab = self.shared_vocab_config.vocab
        self.config = self.shared_vocab_config.config
        self.dropout = self.config.get("dropout", 1)
        self.emb_matrix = self.vocab.emb.lookup
        self.default_vec = np.zeros([self.vocab.emb_length])
        self.char_vocab = self.shared_vocab_config.char_vocab

    def _get_emb(self, idx):
        if idx < self.emb_matrix.shape[0]:
            return self.emb_matrix[idx]
        else:
            return self.default_vec

    @property
    def output_ports(self) -> List[TensorPort]:
        return self._output_ports

    @property
    def training_ports(self) -> List[TensorPort]:
        return self._training_ports

    def preprocess(self, questions: List[QASetting],
                   answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) -> List[XQAAnnotation]:

        if answers is None:
            answers = [None] * len(questions)

        return [self.preprocess_instance(q, a)
                for q, a in zip(questions, answers)]

    def preprocess_instance(self, question: QASetting, answers: Optional[List[Answer]] = None) -> XQAAnnotation:
        has_answers = answers is not None

        q_tokenized, q_ids, _, q_length, s_tokenized, s_ids, _, s_length, \
        word_in_question, token_offsets, answer_spans = prepare_data(
            question, answers, self.vocab, self.config.get("lowercase", False),
            with_answers=has_answers, max_support_length=self.config.get("max_support_length", None))

        emb_support = np.zeros([s_length, self.emb_matrix.shape[1]])
        emb_question = np.zeros([q_length, self.emb_matrix.shape[1]])

        for k in range(len(s_ids)):
            emb_support[k] = self._get_emb(s_ids[k])
        for k in range(len(q_ids)):
            emb_question[k] = self._get_emb(q_ids[k])

        return XQAAnnotation(
            question_tokens=q_tokenized,
            question_ids=q_ids,
            question_length=q_length,
            question_embeddings=emb_question,
            support_tokens=s_tokenized,
            support_ids=s_ids,
            support_length=s_length,
            support_embeddings=emb_support,
            word_in_question=word_in_question,
            token_offsets=token_offsets,
            answer_spans=answer_spans if has_answers else None,
        )

    def create_batch(self, annotations: List[XQAAnnotation], is_eval: bool, with_answers: bool) \
            -> Mapping[TensorPort, np.ndarray]:

        batch_size = len(annotations)

        emb_supports = [a.support_embeddings for a in annotations]
        emb_questions = [a.question_embeddings for a in annotations]

        support_lengths = [a.support_length for a in annotations]
        question_lengths = [a.question_length for a in annotations]
        wiq = [a.word_in_question for a in annotations]
        offsets = [a.token_offsets for a in annotations]

        q_tokenized = [a.question_tokens for a in annotations]
        s_tokenized = [a.support_tokens for a in annotations]

        unique_words, unique_word_lengths, question2unique, support2unique = \
            unique_words_with_chars(q_tokenized, s_tokenized, self.char_vocab)


        output = {
            XQAPorts.unique_word_chars: unique_words,
            XQAPorts.unique_word_char_length: unique_word_lengths,
            XQAPorts.question_words2unique: question2unique,
            XQAPorts.support_words2unique: support2unique,
            XQAPorts.emb_support: stack_and_pad(emb_supports),
            XQAPorts.support_length: support_lengths,
            XQAPorts.emb_question: stack_and_pad(emb_questions),
            XQAPorts.question_length: question_lengths,
            XQAPorts.word_in_question: wiq,
            XQAPorts.keep_prob: 1.0 if is_eval else 1 - self.dropout,
            XQAPorts.is_eval: is_eval,
            XQAPorts.token_char_offsets: offsets,
        }

        if with_answers:
            spans = [a.answer_spans for a in annotations]
            span2question = [i for i in range(batch_size) for _ in spans[i]]
            output.update({
                XQAPorts.answer_span: [span for span_list in spans for span in span_list],
                XQAPorts.correct_start_training: [] if is_eval else [span[0] for span_list in spans for span in
                                                                     span_list],
                XQAPorts.answer2question_training: span2question,
            })

        # we can only numpify in here, because bucketing is not possible prior
        batch = numpify(output, keys=[XQAPorts.unique_word_chars,
                                      XQAPorts.question_words2unique, XQAPorts.support_words2unique,
                                      XQAPorts.word_in_question, XQAPorts.token_char_offsets])
        return batch


class AbstractXQAModelModule(TFModelModule):
    _input_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                    XQAPorts.emb_support, XQAPorts.support_length,
                    # char embedding inputs
                    XQAPorts.unique_word_chars, XQAPorts.unique_word_char_length,
                    XQAPorts.question_words2unique, XQAPorts.support_words2unique,
                    # optional input, provided only during training
                    XQAPorts.answer2question_training, XQAPorts.keep_prob, XQAPorts.is_eval]
    _output_ports = [XQAPorts.start_scores, XQAPorts.end_scores,
                     XQAPorts.span_prediction]
    _training_input_ports = [XQAPorts.start_scores, XQAPorts.end_scores,
                             XQAPorts.answer_span, XQAPorts.answer2question_training]
    _training_output_ports = [Ports.loss]

    @property
    def output_ports(self) -> Sequence[TensorPort]:
        return self._output_ports

    @property
    def input_ports(self) -> Sequence[TensorPort]:
        return self._input_ports

    @property
    def training_input_ports(self) -> Sequence[TensorPort]:
        return self._training_input_ports

    @property
    def training_output_ports(self) -> Sequence[TensorPort]:
        return self._training_output_ports

    def create_output(self, shared_vocab_config, emb_question, question_length,
                      emb_support, support_length,
                      unique_word_chars, unique_word_char_length,
                      question_words2unique, support_words2unique,
                      answer2question, keep_prob, is_eval):
        """extractive QA model
        Args:
            shared_vocab_config: has at least a field config (dict) with keys "rep_dim", "rep_dim_input"
            emb_question: [Q, L_q, N]
            question_length: [Q]
            emb_support: [Q, L_s, N]
            support_length: [Q]
            unique_word_chars
            unique_word_char_length
            question_words2unique
            support_words2unique
            answer2question: [A], only during training, i.e., is_eval=False
            keep_prob: []
            is_eval: []

        Returns:
            start_scores [B, L_s, N], end_scores [B, L_s, N], span_prediction [B, 2]
        """
        raise NotImplementedError('Classes that inherit from AbstractExtractiveQA need to override create_output!')

    def create_training_output(self, shared_resources, start_scores, end_scores, answer_span, answer_to_question):
        return xqa_min_crossentropy_loss(start_scores, end_scores, answer_span, answer_to_question)


def _np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class XQAOutputModule(OutputModule):
    def __init__(self, shared_vocab_confg: SharedResources):
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

            answers.append(Answer(answer, span=(char_start, char_end), score=start_probs[start] * end_probs[end]))

        return answers

    @property
    def input_ports(self) -> List[TensorPort]:
        return [FlatPorts.Prediction.answer_span, XQAPorts.token_char_offsets,
                FlatPorts.Prediction.start_scores, FlatPorts.Prediction.end_scores]


class XQANoScoreOutputModule(OutputModule):
    def __init__(self, shared_vocab_confg: SharedResources):
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

            answers.append(Answer(answer, span=(char_start, char_end), score=1.0))

        return answers

    @property
    def input_ports(self) -> List[TensorPort]:
        return [FlatPorts.Prediction.answer_span, XQAPorts.token_char_offsets]
