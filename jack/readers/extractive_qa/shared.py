"""
This file contains reusable modules for extractive QA models and ports
"""
from typing import NamedTuple

from jack.core import *
from jack.readers.extractive_qa.util import prepare_data
from jack.util import preprocessing
from jack.util.map import numpify
from jack.util.preprocessing import sort_by_tfidf

logger = logging.getLogger(__name__)


class XQAPorts:
    # When feeding embeddings directly
    emb_question = Ports.Input.emb_question
    question_length = Ports.Input.question_length
    emb_support = Ports.Input.emb_support
    support_length = Ports.Input.support_length
    support2question = Ports.Input.support2question

    # but also ids, for char-based embeddings
    word_chars = Ports.Input.word_chars
    word_char_length = Ports.Input.word_char_length
    question_batch_words = Ports.Input.question_batch_words
    support_batch_words = Ports.Input.support_batch_words

    is_eval = Ports.is_eval

    # This feature is model specific and thus, not part of the conventional Ports
    word_in_question = TensorPort(np.float32, [None, None], "word_in_question",
                                  "Represents a 1/0 feature for all context tokens denoting"
                                  " whether it is part of the question or not",
                                  "[Q, support_length]")

    correct_start = TensorPortWithDefault(np.array([0], np.int32), [None], "correct_start",
                                          "Represents the correct start of the span which is given to the"
                                          "model during training for use to predicting end.",
                                          "[A]")

    # output ports
    start_scores = Ports.Prediction.start_scores
    end_scores = Ports.Prediction.end_scores
    answer_span = Ports.Prediction.answer_span
    token_offsets = TensorPort(np.int32, [None, None], "token_offsets",
                               "Character index of tokens in support.",
                               "[S, support_length]")
    selected_support = TensorPort(np.int32, [None], "selected_support",
                                  "Selected support based on TF IDF with question", "[num_support]")

    # ports used during training
    answer2support_training = Ports.Input.answer2support
    answer_span_target = Ports.Target.answer_span


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
                     XQAPorts.question_batch_words, XQAPorts.support_batch_words,
                     # features
                     XQAPorts.word_in_question,
                     # optional, only during training
                     XQAPorts.correct_start, XQAPorts.answer2support_training,
                     XQAPorts.is_eval,
                     # for output module
                     XQAPorts.token_offsets, XQAPorts.selected_support]
    _training_ports = [XQAPorts.answer_span_target, XQAPorts.answer2support_training]

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        # create character vocab + word lengths + char ids per word
        if not self.shared_resources.vocab.frozen:
            preprocessing.fill_vocab((q for q, _ in data), self.shared_resources.vocab,
                                     self.shared_resources.config.get("lowercase", False))
            self.shared_resources.vocab.freeze()
        self.shared_resources.char_vocab = preprocessing.char_vocab_from_vocab(self.shared_resources.vocab)

    def setup(self):
        self._rng = random.Random(1)
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config
        self.embeddings = self.shared_resources.embeddings
        self.__default_vec = np.zeros([self.embeddings.shape[-1]])
        self.char_vocab = self.shared_resources.char_vocab

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

        max_num_support = self.config.get("max_num_support", len(question.support))  # take all per default

        # take max supports by TF-IDF (we subsample to max_num_support in create batch)
        # following https://arxiv.org/pdf/1710.10723.pdf
        if len(question.support) > 1:
            scores = sort_by_tfidf(' '.join(q_tokenized), [' '.join(s) for s in s_tokenized])
            selected_supports = [s_idx for s_idx, _ in scores[:max_num_support]]
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

        max_training_support = self.config.get('max_training_support', 2)
        s_tokenized = []
        support_lengths = []
        wiq = []
        offsets = []
        support2question = []
        support_ids = []
        # aligns with support2question, used in output module to get correct index to original set of supports
        selected_support = []
        all_spans = []
        for i, a in enumerate(annotations):
            all_spans.append([])
            if len(a.support_tokens) > max_training_support > 0 and not is_eval:
                # sample only 2 paragraphs and take first with double probability (the best) to speed
                # things up. Following https://arxiv.org/pdf/1710.10723.pdf
                is_done = False
                any_answer = any(a.answer_spans)
                # sample until there is at least one possible answer (if any)
                while not is_done:
                    selected = self._rng.sample(range(0, len(a.support_tokens) + 1), max_training_support + 1)
                    if 0 in selected and 1 in selected:
                        selected = [s - 1 for s in selected if s > 0]
                    else:
                        selected = [max(0, s - 1) for s in selected[:max_training_support]]
                    is_done = not any_answer or any(a.answer_spans[s] for s in selected)
            else:
                selected = set(range(len(a.support_tokens)))
            for s in selected:
                s_tokenized.append(a.support_tokens[s])
                support_lengths.append(a.support_length[s])
                wiq.append(a.word_in_question[s])
                offsets.append(a.token_offsets[s])
                selected_support.append(a.selected_supports[s])
                support_ids.append(a.support_ids[s])
                support2question.append(i)
                if with_answers:
                    all_spans[-1].append(a.answer_spans[s])

        word_chars, word_lengths, batch_word_ids, batch_vocab, batch_rev_vocab = \
            preprocessing.unique_words_with_chars(q_tokenized + s_tokenized, self.char_vocab)

        emb_support = np.zeros([len(support_lengths), max(support_lengths), self.embeddings.shape[-1]])
        emb_question = np.zeros([len(question_lengths), max(question_lengths), self.embeddings.shape[-1]])

        for i, a in enumerate(annotations):
            for j, t in enumerate(a.question_tokens):
                emb_question[i, j] = self.embeddings.get(t, self.__default_vec)
        for k, s_ids in enumerate(s_tokenized):
            for j, t in enumerate(s_ids):
                emb_support[k, j] = self.embeddings.get(t, self.__default_vec)

        output = {
            XQAPorts.word_chars: word_chars,
            XQAPorts.word_char_length: word_lengths,
            XQAPorts.question_batch_words: batch_word_ids[:len(q_tokenized)],
            XQAPorts.support_batch_words: batch_word_ids[len(q_tokenized):],
            XQAPorts.emb_support: emb_support,
            XQAPorts.support_length: support_lengths,
            XQAPorts.emb_question: emb_question,
            XQAPorts.question_length: question_lengths,
            XQAPorts.word_in_question: wiq,
            XQAPorts.support2question: support2question,
            XQAPorts.is_eval: is_eval,
            XQAPorts.token_offsets: offsets,
            XQAPorts.selected_support: selected_support,
            '__vocab': batch_vocab,
            '__rev_vocab': batch_rev_vocab,
        }

        if with_answers:
            spans = [s for a in all_spans for spans_per_support in a for s in spans_per_support]
            span2support = []
            support_idx = 0
            for a in all_spans:
                for spans_per_support in a:
                    span2support.extend([support_idx] * len(spans_per_support))
                    support_idx += 1
            output.update({
                XQAPorts.answer_span_target: [span for span in spans] if spans else np.zeros([0, 2], np.int32),
                XQAPorts.correct_start: [] if is_eval else [span[0] for span in spans],
                XQAPorts.answer2support_training: span2support,
            })

        # we can only numpify in here, because bucketing is not possible prior
        batch = numpify(output, keys=[XQAPorts.word_chars,
                                      XQAPorts.question_batch_words, XQAPorts.support_batch_words,
                                      XQAPorts.word_in_question, XQAPorts.token_offsets])
        return batch


def _np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_answer_and_span(question, doc_idx, start, end, token_offsets, selected_support):
    doc_idx = selected_support[doc_idx]
    char_start = token_offsets[start]
    char_end = token_offsets[end + 1] if end < len(token_offsets) - 1 else len(question.support[doc_idx])
    answer = question.support[doc_idx][char_start: char_end]
    answer = answer.rstrip()
    char_end = char_start + len(answer)
    return answer, doc_idx, (char_start, char_end)


class XQAOutputModule(OutputModule):
    @property
    def input_ports(self) -> List[TensorPort]:
        return [XQAPorts.answer_span, XQAPorts.token_offsets,
                XQAPorts.selected_support, XQAPorts.support2question,
                XQAPorts.start_scores, XQAPorts.end_scores]

    def __call__(self, questions, tensors: Mapping[TensorPort, np.ndarray]) -> Sequence[Sequence[Answer]]:
        """Produces top-k answers for each question."""
        tensors = TensorPortTensors(tensors)
        topk = tensors.answer_span.shape[0] // len(questions)
        all_answers = []
        for n, q in enumerate(questions):
            answers = []
            doc_idx_map = [i for i, q_id in enumerate(tensors.support2question) if q_id == n]
            for j in range(topk):
                i = n * topk + j
                doc_idx, start, end = tensors.answer_span[i]
                score = (_np_softmax(tensors.start_scores[doc_idx_map[doc_idx]])[start] *
                         _np_softmax(tensors.end_scores[doc_idx_map[doc_idx]])[end])
                answer, doc_idx, span = get_answer_and_span(
                    q, doc_idx, start, end, tensors.token_offsets[doc_idx_map[doc_idx]],
                    [i for q_id, i in zip(tensors.support2question, tensors.selected_support) if q_id == n])
                answers.append(Answer(answer, span=span, doc_idx=doc_idx, score=score))
            all_answers.append(answers)

        return all_answers

