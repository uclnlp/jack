"""
This file contains reusable modules for extractive QA models and ports
"""
from typing import NamedTuple

from jack.core import *
from jack.readers.extractive_qa.util import prepare_data
from jack.tfutil.xqa import xqa_min_crossentropy_loss
from jack.util import preprocessing
from jack.util.map import numpify


class XQAPorts:
    # When feeding embeddings directly
    emb_question = Ports.Input.embedded_question
    question_length = Ports.Input.question_length
    emb_support = Ports.Input.embedded_support
    support_length = Ports.Input.support_length
    support2question = Ports.Input.support2question

    # but also ids, for char-based embeddings
    word_chars = TensorPort(tf.int32, [None, None], "question_chars",
                            "Represents questions using symbol vectors",
                            "[U, max_num_chars]")
    word_length = TensorPort(tf.int32, [None], "question_char_length",
                             "Represents questions using symbol vectors",
                             "[U]")
    question_words = TensorPort(tf.int32, [None, None], "question_words2unique",
                                "Represents support using symbol vectors indexing defined word chars.",
                                "[batch_size, max_num_question_tokens]")
    support_words = TensorPort(tf.int32, [None, None], "support_words2unique",
                               "Represents support using symbol vectors indexing defined word chars",
                               "[batch_size, max_num_support_tokens, max]")

    keep_prob = Ports.keep_prob
    is_eval = Ports.is_eval

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
    start_scores = Ports.Prediction.start_scores
    end_scores = Ports.Prediction.end_scores
    span_prediction = Ports.Prediction.answer_span
    token_offsets = TensorPort(tf.int32, [None, None, 2], "token_offsets",
                               "Document and character index of tokens in support.",
                               "[S, support_length, 2]")

    # ports used during training
    answer2support_training = Ports.Input.answer2support
    answer_span = Ports.Target.answer_span


XQAAnnotation = NamedTuple('XQAAnnotation', [
    ('question_tokens', List[str]),
    ('question_ids', List[int]),
    ('question_length', int),
    ('question_embeddings', List[np.ndarray]),
    ('support_tokens', List[List[str]]),
    ('support_ids', List[List[int]]),
    ('support_length', List[int]),
    ('support_embeddings', List[List[np.ndarray]]),
    ('word_in_question', List[List[float]]),
    ('token_offsets', List[List[int]]),
    ('answer_spans', Optional[List[List[Tuple[int, int]]]]),
])


class XQAInputModule(OnlineInputModule[XQAAnnotation]):
    _output_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                     XQAPorts.emb_support, XQAPorts.support_length,
                     XQAPorts.support2question,
                     # char
                     XQAPorts.word_chars, XQAPorts.word_length,
                     XQAPorts.question_words, XQAPorts.support_words,
                     # features
                     XQAPorts.word_in_question,
                     # optional, only during training
                     XQAPorts.correct_start_training, XQAPorts.answer2support_training,
                     XQAPorts.keep_prob, XQAPorts.is_eval,
                     # for output module
                     XQAPorts.token_offsets]
    _training_ports = [XQAPorts.answer_span, XQAPorts.answer2support_training]

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
        if self.vocab.emb is None:
            logger.error("XQAInputModule needs vocabulary setup from pre-trained embeddings."
                         "Make sure to set vocab_from_embeddings=True.")
            sys.exit(1)
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

        return [self.preprocess_instance(q, a) for q, a in zip(questions, answers)]

    def preprocess_instance(self, question: QASetting, answers: Optional[List[Answer]] = None) -> XQAAnnotation:
        has_answers = answers is not None

        q_tokenized, q_ids, _, q_length, s_tokenized, s_ids, _, s_length, \
        word_in_question, token_offsets, answer_spans = prepare_data(
            question, answers, self.vocab, self.config.get("lowercase", False),
            with_answers=has_answers, max_support_length=self.config.get("max_support_length", None))

        emb_support = []
        emb_question = []

        for k in range(len(s_ids)):
            emb_support.append([])
            for j in range(len(s_ids[k])):
                emb_support[-1].append(self._get_emb(s_ids[k][j]))
        for k in range(len(q_ids)):
            emb_question.append(self._get_emb(q_ids[k]))

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

        q_tokenized = [a.question_tokens for a in annotations]
        emb_questions = [np.stack(a.question_embeddings) for a in annotations]
        question_lengths = [a.question_length for a in annotations]

        s_tokenized = [ts for a in annotations for ts in a.support_tokens]
        support_lengths = [l for a in annotations for l in a.support_length]
        emb_supports = [np.stack(s_embs) for a in annotations for s_embs in a.support_embeddings]
        wiq = [wiq for a in annotations for wiq in a.word_in_question]
        offsets = [offsets for a in annotations for offsets in a.token_offsets]
        support2question = [i for i, a in enumerate(annotations) for _ in a.support_tokens]

        unique_words, unique_word_lengths, question2unique, support2unique = \
            preprocessing.unique_words_with_chars(q_tokenized, s_tokenized, self.char_vocab)

        output = {
            XQAPorts.word_chars: unique_words,
            XQAPorts.word_length: unique_word_lengths,
            XQAPorts.question_words: question2unique,
            XQAPorts.support_words: support2unique,
            XQAPorts.emb_support: preprocessing.stack_and_pad(emb_supports),
            XQAPorts.support_length: support_lengths,
            XQAPorts.emb_question: preprocessing.stack_and_pad(emb_questions),
            XQAPorts.question_length: question_lengths,
            XQAPorts.word_in_question: wiq,
            XQAPorts.support2question: support2question,
            XQAPorts.keep_prob: 1.0 if is_eval else 1 - self.dropout,
            XQAPorts.is_eval: is_eval,
            XQAPorts.token_offsets: offsets,
        }

        if with_answers:
            spans = [s for a in annotations for spans_per_support in a.answer_spans for s in spans_per_support]
            span2support = []
            support_idx = 0
            for a in annotations:
                for spans_per_support in a.answer_spans:
                    span2support.extend([support_idx] * len(spans_per_support))
                    support_idx += 1
            output.update({
                XQAPorts.answer_span: [span for span in spans],
                XQAPorts.correct_start_training: [] if is_eval else [span[0] for span in spans],
                XQAPorts.answer2support_training: span2support,
            })

        # we can only numpify in here, because bucketing is not possible prior
        batch = numpify(output, keys=[XQAPorts.word_chars,
                                      XQAPorts.question_words, XQAPorts.support_words,
                                      XQAPorts.word_in_question, XQAPorts.token_offsets])
        return batch


class AbstractXQAModelModule(TFModelModule):
    _input_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                    XQAPorts.emb_support, XQAPorts.support_length,
                    XQAPorts.support2question,
                    # char embedding inputs
                    XQAPorts.word_chars, XQAPorts.word_length,
                    XQAPorts.question_words, XQAPorts.support_words,
                    # optional input, provided only during training
                    XQAPorts.answer2support_training, XQAPorts.keep_prob, XQAPorts.is_eval]
    _output_ports = [XQAPorts.start_scores, XQAPorts.end_scores,
                     XQAPorts.span_prediction]
    _training_input_ports = [XQAPorts.start_scores, XQAPorts.end_scores,
                             XQAPorts.answer_span, XQAPorts.answer2support_training, XQAPorts.support2question]
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

    def create_output(self, shared_resources, emb_question, question_length,
                      emb_support, support_length, support2question,
                      unique_word_chars, unique_word_char_length,
                      question_words2unique, support_words2unique,
                      answer2support, keep_prob, is_eval):
        """extractive QA model
        Args:
            shared_resources: has at least a field config (dict) with keys "rep_dim", "rep_dim_input"
            emb_question: [Q, L_q, N]
            question_length: [Q]
            emb_support: [Q, L_s, N]
            support_length: [Q]
            unique_word_chars
            unique_word_char_length
            question_words2unique
            support_words2unique
            answer2support: [A], only during training, i.e., is_eval=False
            keep_prob: []
            is_eval: []

        Returns:
            start_scores [B, L_s, N], end_scores [B, L_s, N], span_prediction [B, 2]
        """
        raise NotImplementedError('Classes that inherit from AbstractExtractiveQA need to override create_output!')

    def create_training_output(
            self, shared_resources, start_scores, end_scores, answer_span, answer2support, support2question):
        return xqa_min_crossentropy_loss(start_scores, end_scores, answer_span, answer2support, support2question)


def _np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_answer_and_span(question, support_length, span_prediction, token_offsets):
    doc_idx, start, end = span_prediction
    char_start = token_offsets[start]
    char_end = token_offsets[end]
    answer = question.support[doc_idx][char_start: char_end]
    answer = answer.rstrip()
    char_end = char_start + len(answer)
    return answer, doc_idx, (char_start, char_end)


class XQAOutputModule(OutputModule):
    def __init__(self, shared_resources):
        self.beam_size = shared_resources.config.get("beam_size", 1)

    def __call__(self, questions, span_prediction, token_offsets, support_length,
                 start_scores, end_scores):
        all_answers = []
        for k, q in enumerate(questions):
            answers = []
            for j in range(self.beam_size):
                i = k * self.beam_size + j
                _, start, end = span_prediction[i]
                answer, doc_idx, span = get_answer_and_span(q, support_length[i],
                                                            span_prediction[i],
                                                            token_offsets[i])

                start_probs = _np_softmax(start_scores[i])
                end_probs = _np_softmax(end_scores[i])

                answers.append(Answer(answer, span=span, doc_idx=doc_idx,
                                      score=start_probs[start] * end_probs[end]))
            all_answers.append(answers)

        return all_answers

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.answer_span, XQAPorts.token_offsets,
                XQAPorts.support_length,
                Ports.Prediction.start_scores, Ports.Prediction.end_scores]

