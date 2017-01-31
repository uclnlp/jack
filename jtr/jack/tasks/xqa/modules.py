"""
This file contains modules for extractive QA models that have an additional
"""

import random

import jtr
from jtr.jack import *
from jtr.jack.fun import model_module_factory, model_module
from jtr.jack.tasks.xqa.util import token_to_char_offsets
from jtr.preprocess.batch import get_batches, GeneratorWithRestart
from jtr.preprocess.map import deep_map, numpify
from jtr.jack.tasks.xqa.fast_qa import fastqa_model


class XqaPorts:
    """
    It is good practice define all ports needed for a single model jointly, to get an overview
    """

    # We feed embeddings directly
    emb_question = FlatPorts.Misc.embedded_question
    question_length = FlatPorts.Input.question_length
    emb_support = FlatPorts.Misc.embedded_support
    support_length = FlatPorts.Input.support_length
    keep_prob = Ports.Input.keep_prob
    is_eval = Ports.Input.is_eval

    # This feature is model specific and thus, not part of the conventional Ports
    word_in_question = TensorPort(tf.float32, [None, None], "word_in_question_feature",
                                  "Represents a 1/0 feature for all context tokens denoting"
                                  " whether it is part of the question or not",
                                  "[Q, support_length]")

    # output ports
    start_scores = FlatPorts.Prediction.start_scores
    end_scores = FlatPorts.Prediction.end_scores
    span_prediction = FlatPorts.Prediction.answer_span

    # port provided by input but only consumed by output module -> needed to transform token to char offsets
    token_char_offsets = TensorPort(tf.int32, [None, None], "token_char_offsets",
                                    "Character offsets of tokens in support.",
                                    "[S, support_length]")


    # ports used during training
    answer_to_question = FlatPorts.Input.answer_to_question
    answer_span = FlatPorts.Target.answer_span


class XqaWiqInputModule(InputModule):

    def __init__(self, shared_vocab_config):
        assert isinstance(shared_vocab_config, SharedVocabAndConfig), \
            "shared_resources for XqaWiqInputModule must be an instance of SharedVocabAndConfig"

        self.shared_vocab_config = shared_vocab_config
        vocab=shared_vocab_config.vocab
        config=shared_vocab_config.config
        self.batch_size = config["batch_size"]
        self.dropout = config.get("dropout", 1)
        self._rng = random.Random(config.get("seed", 123))
        self.emb_matrix = vocab.emb.lookup
        self.default_vec = np.zeros([vocab.emb_length])

    def _get_emb(self, idx):
        if idx < self.emb_matrix.shape[0]:
            return self.emb_matrix[idx]
        else:
            return self.default_vec

    @property
    def output_ports(self) -> List[TensorPort]:
        return [XqaPorts.emb_question, XqaPorts.question_length,
                XqaPorts.emb_support, XqaPorts.support_length,
                XqaPorts.word_in_question, XqaPorts.keep_prob, XqaPorts.is_eval,
                XqaPorts.token_char_offsets]

    @property
    def training_ports(self) -> List[TensorPort]:
        return [XqaPorts.answer_span, XqaPorts.answer_to_question]

    def setup_from_data(self, data: List[Tuple[Question, List[Answer]]]) -> SharedResources:
        # Assumes that vocab and embeddings are given during creation
        return self.shared_vocab_config

    def setup(self, shared_resources: SharedResources):
        assert isinstance(shared_resources, SharedVocabAndConfig), \
            "shared_resources for XqaWiqInputModule must be an instance of SharedVocabAndConfig"
        self.shared_vocab_config = shared_resources

    def dataset_generator(self, dataset: List[Tuple[Question, List[Answer]]], is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus = {"support": [], "support_lengths": [], "question": [], "question_lengths": []}

        for input, answers in dataset:
            corpus["support"].append(input.support[0])
            corpus["question"].append(input.question)

        corpus = deep_map(corpus, jtr.preprocess.map.tokenize, ['question', 'support'])
        word_in_question = []

        token_offsets = []
        answer_spans = []
        for i, (q, s) in enumerate(zip(corpus["question"], corpus["support"])):
            input, answers = dataset[i]
            corpus["support_lengths"].append(len(s))
            corpus["question_lengths"].append(len(q))

            # char to token offsets
            offsets = token_to_char_offsets(input.support[0], s)
            token_offsets.append(offsets)
            spans = []
            for a in answers:
                start = 0
                while offsets[start] < a.span[0]:
                    start += 1
                end = start
                while end < len(offsets) and offsets[end] < a.span[1]:
                    end += 1
                if (start, end) not in spans:
                    spans.append((start, end))
            answer_spans.append(spans)

            # word in question feature
            wiq = []
            for token in s:
                wiq.append(float(token in q))
            word_in_question.append(wiq)

        emb_supports = np.zeros([self.batch_size, max(corpus["support_lengths"]), self.emb_matrix.shape[1]])
        emb_questions = np.zeros([self.batch_size, max(corpus["question_lengths"]), self.emb_matrix.shape[1]])

        corpus_ids = deep_map(corpus, self.shared_vocab_config.vocab, ['question', 'support'])

        def batch_generator():
            todo = list(range(len(corpus_ids["question"])))
            self._rng.shuffle(todo)
            while todo:
                supports = list()
                support_lengths = list()
                questions = list()
                question_lengths = list()
                wiq = list()
                spans = list()
                span2question = []
                offsets = []

                # we have to create batches here and cannot precompute them because of the batch-specific wiq feature
                for i, j in enumerate(todo[:self.batch_size]):
                    supports.append(corpus_ids["support"][j])
                    support_lengths.append(corpus["support_lengths"][j])
                    questions.append(corpus_ids["question"][j])
                    question_lengths.append(corpus["question_lengths"][j])
                    spans.extend(answer_spans[j])
                    span2question.extend(i for _ in answer_spans[j])
                    wiq.append(word_in_question[j])
                    offsets.append(token_offsets[j])

                for i in range(len(supports)):
                    for j in range(len(supports[i])):
                        emb_supports[i, j] = self._get_emb(supports[i][j])
                for i in range(len(questions)):
                    for j in range(len(questions[i])):
                        emb_questions[i, j] = self._get_emb(questions[i][j])

                output = {
                    XqaPorts.emb_support: emb_supports[:len(supports),:max(support_lengths),:],
                    XqaPorts.support_length: support_lengths,
                    XqaPorts.emb_question: emb_questions[:len(questions),:max(question_lengths),:],
                    XqaPorts.question_length: question_lengths,
                    XqaPorts.word_in_question: wiq,
                    XqaPorts.answer_span: spans,
                    XqaPorts.answer_to_question: span2question,
                    XqaPorts.keep_prob: 0.0 if is_eval else 1 - self.dropout,
                    XqaPorts.is_eval: is_eval,
                    XqaPorts.token_char_offsets: offsets
                }

                # we can only numpify in here, because bucketing is not possible prior
                batch = numpify(output, keys=[XqaPorts.word_in_question, XqaPorts.token_char_offsets])
                yield batch

        return GeneratorWithRestart(batch_generator)

    def __call__(self, inputs: List[Question]) -> Mapping[TensorPort, np.ndarray]:
        supports = list()
        questions = list()
        for input in inputs:
            supports.append(input.support[0])
            questions.append(input.question)

        supports = deep_map(supports, self.shared_vocab_config.vocab)
        questions = deep_map(questions, self.shared_vocab_config.vocab)

        word_in_question = []

        for q, s in zip(questions, supports):
            wiq = []
            for token in s:
                wiq.append(float(token in q))
            word_in_question.append(wiq)

        supports = deep_map(supports, self._get_emb)
        questions = deep_map(questions, self._get_emb)

        output = {
            XqaPorts.emb_support: supports,
            XqaPorts.emb_question: questions,
            XqaPorts.word_in_question: word_in_question,
            XqaPorts.keep_prob: 0.0
        }

        return numpify(output)


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
        return [XqaPorts.span_prediction, XqaPorts.token_char_offsets]

    def setup(self, shared_resources):
        self.vocab = shared_resources.vocab


def xqa_min_crossentropy_loss(shared_resources, start_scores, end_scores, answer_span, answer_to_question) -> List[tf.Tensor]:
    start, end = [tf.squeeze(t, 1) for t in tf.split(1, 2, answer_span)]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(start_scores, start) + \
                   tf.nn.sparse_softmax_cross_entropy_with_logits(end_scores, end)

    loss = tf.segment_min(loss, answer_to_question)
    return [tf.reduce_mean(loss)]


xqa_wiq_with_min_crossentropy_loss =\
    model_module_factory(input_ports=[XqaPorts.emb_question, XqaPorts.question_length,
                                      XqaPorts.emb_support, XqaPorts.support_length,
                                      XqaPorts.word_in_question,
                                      XqaPorts.keep_prob, XqaPorts.is_eval],
                         output_ports=[XqaPorts.start_scores, XqaPorts.end_scores, XqaPorts.span_prediction],
                         training_input_ports=[XqaPorts.start_scores, XqaPorts.end_scores, XqaPorts.answer_span,
                                               XqaPorts.answer_to_question],
                         training_output_ports=[Ports.loss],
                         training_function=xqa_min_crossentropy_loss)
