"""
This file contains reusable modules for extractive QA models and ports
"""
import sys
from collections import defaultdict
from typing import NamedTuple

import progressbar

from jack.core import *
from jack.readers.extractive_qa.util import prepare_data
from jack.util import preprocessing
from jack.util.map import numpify

logger = logging.getLogger(__name__)


class XQAPorts:
    # When feeding embeddings directly
    emb_question = Ports.Input.emb_question
    question_length = Ports.Input.question_length
    emb_support = Ports.Input.emb_support
    support_length = Ports.Input.support_length
    support2question = Ports.Input.support2question

    # but also ids, for char-based embeddings
    word_chars = TensorPort(tf.int32, [None, None], "word_chars",
                            "Represents questions using symbol vectors",
                            "[U, max_num_chars]")
    word_char_length = TensorPort(tf.int32, [None], "word_char_length",
                                  "Represents questions using symbol vectors",
                                  "[U]")
    question_words = TensorPort(tf.int32, [None, None], "question_words",
                                "Represents support using symbol vectors indexing defined word chars.",
                                "[batch_size, max_num_question_tokens]")
    support_words = TensorPort(tf.int32, [None, None], "support_words",
                               "Represents support using symbol vectors indexing defined word chars",
                               "[batch_size, max_num_support_tokens, max]")

    is_eval = Ports.is_eval

    # This feature is model specific and thus, not part of the conventional Ports
    word_in_question = TensorPort(tf.float32, [None, None], "word_in_question",
                                  "Represents a 1/0 feature for all context tokens denoting"
                                  " whether it is part of the question or not",
                                  "[Q, support_length]")

    correct_start = TensorPortWithDefault(np.array([0], np.int32), tf.int32, [None], "correct_start",
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
    selected_support = TensorPort(tf.int32, [None], "selected_support",
                                  "Selected support based on TF IDF with question", "[num_support]")

    # ports used during training
    answer2support_training = Ports.Input.answer2support
    answer_span = Ports.Target.answer_span


XQAAnnotation = NamedTuple('XQAAnnotation', [
    ('question_tokens', List[str]),
    ('question_ids', List[int]),
    ('question_length', int),
    ('support_tokens', List[List[str]]),
    ('support_ids', List[List[int]]),
    ('support_length', List[int]),
    ('word_in_question', List[List[float]]),
    ('token_offsets', List[List[int]]),
    ('answer_spans', Optional[List[List[Tuple[int, int]]]]),
    ('selected_supports', Optional[List[int]]),
])


class XQAInputModule(OnlineInputModule[XQAAnnotation]):
    _output_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                     XQAPorts.emb_support, XQAPorts.support_length,
                     XQAPorts.support2question,
                     # char
                     XQAPorts.word_chars, XQAPorts.word_char_length,
                     XQAPorts.question_words, XQAPorts.support_words,
                     # features
                     XQAPorts.word_in_question,
                     # optional, only during training
                     XQAPorts.correct_start, XQAPorts.answer2support_training,
                     XQAPorts.is_eval,
                     # for output module
                     XQAPorts.token_offsets, XQAPorts.selected_support]
    _training_ports = [XQAPorts.answer_span, XQAPorts.answer2support_training]

    def __init__(self, shared_vocab_config):
        assert isinstance(shared_vocab_config, SharedResources), \
            "shared_resources for FastQAInputModule must be an instance of SharedResources"
        self.shared_resources = shared_vocab_config
        self._rng = random.Random(1)

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        # create character vocab + word lengths + char ids per word
        self.shared_resources.char_vocab = preprocessing.char_vocab_from_vocab(self.shared_resources.vocab)

    def setup(self):
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config
        if self.vocab.emb is None:
            logger.error("XQAInputModule needs vocabulary setup from pre-trained embeddings."
                         "Make sure to set vocab_from_embeddings=True.")
            sys.exit(1)
        self.emb_matrix = self.vocab.emb.lookup
        self.default_vec = np.zeros([self.vocab.emb_length])
        self.char_vocab = self.shared_resources.char_vocab

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
        preprocessed = []
        if len(questions) > 1000:
            bar = progressbar.ProgressBar(
                max_value=len(questions),
                widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') '])
            for q, a in bar(zip(questions, answers)):
                preprocessed.append(self.preprocess_instance(q, a))
        else:
            for q, a in zip(questions, answers):
                preprocessed.append(self.preprocess_instance(q, a))

        return preprocessed

    def preprocess_instance(self, question: QASetting, answers: Optional[List[Answer]] = None) -> XQAAnnotation:
        has_answers = answers is not None

        q_tokenized, q_ids, _, q_length, s_tokenized, s_ids, _, s_length, \
        word_in_question, token_offsets, answer_spans = prepare_data(
            question, answers, self.vocab, self.config.get("lowercase", False),
            with_answers=has_answers, max_support_length=self.config.get("max_support_length", None))

        max_num_support = self.config.get("max_num_support")  # take all per default
        if max_num_support is not None and len(question.support) > max_num_support:
            # take 2 * the number of max supports by TF-IDF (we subsample to max_num_support in create batch)
            # following https://arxiv.org/pdf/1710.10723.pdf
            q_freqs = defaultdict(float)
            freqs = defaultdict(float)
            for w, i in zip(q_tokenized, q_ids):
                if w.isalnum():
                    q_freqs[i] += 1.0
                    freqs[i] += 1.0
            d_freqs = []
            for i, s in enumerate(s_ids):
                d_freqs.append(defaultdict(float))
                for j in s:
                    freqs[j] += 1.0
                    d_freqs[-1][j] += 1.0
            scores = []
            for i, d_freq in enumerate(d_freqs):
                score = sum(v / freqs[k] * d_freq.get(k, 0.0) / freqs[k] for k, v in q_freqs.items())
                scores.append((i, score))

            selected_supports = [s_idx for s_idx, _ in sorted(scores, key=lambda x: -x[1])[:max_num_support]]
            s_tokenized = [s_tokenized[s_idx] for s_idx in selected_supports]
            s_ids = [s_ids[s_idx] for s_idx in selected_supports]
            s_length = [s_length[s_idx] for s_idx in selected_supports]
            word_in_question = [word_in_question[s_idx] for s_idx in selected_supports]
            token_offsets = [token_offsets[s_idx] for s_idx in selected_supports]
            answer_spans = [answer_spans[s_idx] for s_idx in selected_supports]
        else:
            selected_supports = list(range(len(question.support)))

        return XQAAnnotation(
            question_tokens=q_tokenized,
            question_ids=q_ids,
            question_length=q_length,
            support_tokens=s_tokenized,
            support_ids=s_ids,
            support_length=s_length,
            word_in_question=word_in_question,
            token_offsets=token_offsets,
            answer_spans=answer_spans if has_answers else None,
            selected_supports=selected_supports,
        )

    def create_batch(self, annotations: List[XQAAnnotation], is_eval: bool, with_answers: bool) \
            -> Mapping[TensorPort, np.ndarray]:

        q_tokenized = [a.question_tokens for a in annotations]
        question_lengths = [a.question_length for a in annotations]

        max_num_support = self.config.get("max_num_support")  # take all per default
        s_tokenized = []
        support_lengths = []
        wiq = []
        offsets = []
        support2question = []
        # aligns with support2question, used in output module to get correct index to original set of supports
        selected_support = []
        for j, a in enumerate(annotations):
            if max_num_support is not None and len(a.support_tokens) > max(1, max_num_support // 2) and not is_eval:
                # always take first (the best) and sample from rest during training, only consider half to speed
                # things up. Following https://arxiv.org/pdf/1710.10723.pdf we sample half during training
                selected = self._rng.sample(range(1, len(a.support_tokens)), max(1, max_num_support // 2) - 1)
                selected = set([0] + selected)
            else:
                selected = set(range(len(a.support_tokens)))
            for s in selected:
                s_tokenized.append(a.support_tokens[s])
                support_lengths.append(a.support_length[s])
                wiq.append(a.word_in_question[s])
                offsets.append(a.token_offsets[s])
                selected_support.append(a.selected_supports[s])
                support2question.append(j)

        word_chars, word_lengths, word_ids, vocab, rev_vocab = \
            preprocessing.unique_words_with_chars(q_tokenized + s_tokenized, self.char_vocab)

        emb_support = np.zeros([len(support_lengths), max(support_lengths), self.vocab.emb_length])
        emb_question = np.zeros([len(question_lengths), max(question_lengths), self.vocab.emb_length])

        k = 0
        for i, a in enumerate(annotations):
            for j, q_id in enumerate(a.question_ids):
                emb_question[i, j] = self._get_emb(q_id)
            for s_ids in a.support_ids:
                for j, s_id in enumerate(s_ids):
                    emb_support[k, j] = self._get_emb(s_id)
                k += 1

        output = {
            XQAPorts.word_chars: word_chars,
            XQAPorts.word_char_length: word_lengths,
            XQAPorts.question_words: word_ids[:len(q_tokenized)],
            XQAPorts.support_words: word_ids[len(q_tokenized):],
            XQAPorts.emb_support: emb_support,
            XQAPorts.support_length: support_lengths,
            XQAPorts.emb_question: emb_question,
            XQAPorts.question_length: question_lengths,
            XQAPorts.word_in_question: wiq,
            XQAPorts.support2question: support2question,
            XQAPorts.is_eval: is_eval,
            XQAPorts.token_offsets: offsets,
            XQAPorts.selected_support: selected_support,
            '__vocab': vocab,
            '__rev_vocab': rev_vocab,
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
                XQAPorts.correct_start: [] if is_eval else [span[0] for span in spans],
                XQAPorts.answer2support_training: span2support,
            })

        # we can only numpify in here, because bucketing is not possible prior
        batch = numpify(output, keys=[XQAPorts.word_chars,
                                      XQAPorts.question_words, XQAPorts.support_words,
                                      XQAPorts.word_in_question, XQAPorts.token_offsets])
        return batch


def _np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_answer_and_span(question, doc_idx, start, end, token_offsets, selected_support):
    doc_idx = selected_support[doc_idx]
    char_start = token_offsets[start]
    if end < len(token_offsets) - 1:
        char_end = token_offsets[end + 1]
    else:
        char_end = len(question.support[doc_idx])
    answer = question.support[doc_idx][char_start: char_end]
    answer = answer.rstrip()
    char_end = char_start + len(answer)
    return answer, doc_idx, (char_start, char_end)


class XQAOutputModule(OutputModule):
    def __init__(self, shared_resources):
        self.beam_size = shared_resources.config.get("beam_size", 1)

    def __call__(self, questions, span_prediction,
                 token_offsets, selected_support, support2question,
                 start_scores, end_scores):
        all_answers = []
        for k, q in enumerate(questions):
            answers = []
            doc_idx_map = [i for i, q_id in enumerate(support2question) if q_id == k]
            for j in range(self.beam_size):
                i = k * self.beam_size + j
                doc_idx, start, end = span_prediction[i]
                score = start_scores[doc_idx_map[doc_idx], start]
                answer, doc_idx, span = get_answer_and_span(
                    q, doc_idx, start, end, token_offsets[doc_idx_map[doc_idx]],
                    [i for q_id, i in zip(support2question, selected_support) if q_id == k])
                answers.append(Answer(answer, span=span, doc_idx=doc_idx, score=score))
            all_answers.append(answers)

        return all_answers

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.answer_span, XQAPorts.token_offsets,
                XQAPorts.selected_support, XQAPorts.support2question,
                Ports.Prediction.start_scores, Ports.Prediction.end_scores]
