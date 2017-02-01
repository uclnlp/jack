"""
This file contains FastQA specific modules and ports
"""

import random

import jtr
from jtr.jack import *
from jtr.jack.fun import model_module_factory, no_shared_resources
from jtr.jack.tasks.xqa.shared import XqaPorts
from jtr.jack.tasks.xqa.util import token_to_char_offsets
from jtr.jack.tf_fun.xqa import xqa_min_crossentropy_loss
from jtr.preprocess.batch import GeneratorWithRestart
from jtr.preprocess.map import deep_map, numpify


class FastQAPorts:
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
    token_char_offsets = XqaPorts.token_char_offsets

    # ports used during training
    answer_to_question = FlatPorts.Input.answer_to_question
    answer_span = FlatPorts.Target.answer_span


# FastQA model module factory method, like fastqa.model.fastqa_model
fastqa_with_min_crossentropy_loss =\
    model_module_factory(input_ports=[FastQAPorts.emb_question, FastQAPorts.question_length,
                                      FastQAPorts.emb_support, FastQAPorts.support_length,
                                      FastQAPorts.word_in_question,
                                      FastQAPorts.keep_prob, FastQAPorts.is_eval],
                         output_ports=[FastQAPorts.start_scores, FastQAPorts.end_scores, FastQAPorts.span_prediction],
                         training_input_ports=[FastQAPorts.start_scores, FastQAPorts.end_scores, FastQAPorts.answer_span,
                                               FastQAPorts.answer_to_question],
                         training_output_ports=[Ports.loss],
                         training_function=no_shared_resources(xqa_min_crossentropy_loss))


class FastQAInputModule(InputModule):

    def __init__(self, shared_vocab_config):
        assert isinstance(shared_vocab_config, SharedVocabAndConfig), \
            "shared_resources for FastQAInputModule must be an instance of SharedVocabAndConfig"

        self.shared_vocab_config = shared_vocab_config
        vocab = shared_vocab_config.vocab
        config = shared_vocab_config.config
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
        return [FastQAPorts.emb_question, FastQAPorts.question_length,
                FastQAPorts.emb_support, FastQAPorts.support_length,
                FastQAPorts.word_in_question, FastQAPorts.keep_prob, FastQAPorts.is_eval,
                FastQAPorts.token_char_offsets]

    @property
    def training_ports(self) -> List[TensorPort]:
        return [FastQAPorts.answer_span, FastQAPorts.answer_to_question]

    def setup_from_data(self, data: List[Tuple[Question, List[Answer]]]) -> SharedResources:
        # Assumes that vocab and embeddings are given during creation
        return self.shared_vocab_config

    def setup(self, shared_resources: SharedResources):
        assert isinstance(shared_resources, SharedVocabAndConfig), \
            "shared_resources for FastQAInputModule must be an instance of SharedVocabAndConfig"
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
                while end+1 < len(offsets) and offsets[end+1] < a.span[1]:
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
                support_lengths = list()
                question_lengths = list()
                wiq = list()
                spans = list()
                span2question = []
                offsets = []

                # we have to create batches here and cannot precompute them because of the batch-specific wiq feature
                for i, j in enumerate(todo[:self.batch_size]):
                    for k in range(len(corpus_ids["support"][j])):
                        emb_supports[i, k] = self._get_emb(corpus_ids["support"][j][k])
                    for k in range(len(corpus_ids["question"][j])):
                        emb_supports[i, k] = self._get_emb(corpus_ids["question"][j][k])
                    support_lengths.append(corpus["support_lengths"][j])
                    question_lengths.append(corpus["question_lengths"][j])
                    spans.extend(answer_spans[j])
                    span2question.extend(i for _ in answer_spans[j])
                    wiq.append(word_in_question[j])
                    offsets.append(token_offsets[j])

                output = {
                    FastQAPorts.emb_support: emb_supports[:len(support_lengths), :max(support_lengths), :],
                    FastQAPorts.support_length: support_lengths,
                    FastQAPorts.emb_question: emb_questions[:len(question_lengths), :max(question_lengths), :],
                    FastQAPorts.question_length: question_lengths,
                    FastQAPorts.word_in_question: wiq,
                    FastQAPorts.answer_span: spans,
                    FastQAPorts.answer_to_question: span2question,
                    FastQAPorts.keep_prob: 0.0 if is_eval else 1 - self.dropout,
                    FastQAPorts.is_eval: is_eval,
                    FastQAPorts.token_char_offsets: offsets
                }

                # we can only numpify in here, because bucketing is not possible prior
                batch = numpify(output, keys=[FastQAPorts.word_in_question, FastQAPorts.token_char_offsets])
                todo = todo[self.batch_size:]
                yield batch

        return GeneratorWithRestart(batch_generator)

    def __call__(self, inputs: List[Question]) -> Mapping[TensorPort, np.ndarray]:
        corpus = {"support": [], "support_lengths": [], "question": [], "question_lengths": []}
        for input in inputs:
            corpus["support"].append(input.support[0])
            corpus["question"].append(input.question)

        corpus = deep_map(corpus, jtr.preprocess.map.tokenize, ['question', 'support'])
        word_in_question = []

        token_offsets = []
        for q, s, input in enumerate(zip(corpus["question"], corpus["support"], inputs)):
            corpus["support_lengths"].append(len(s))
            corpus["question_lengths"].append(len(q))

            # char to token offsets
            offsets = token_to_char_offsets(input.support[0], s)
            token_offsets.append(offsets)

            # word in question feature
            wiq = []
            for token in s:
                wiq.append(float(token in q))
            word_in_question.append(wiq)

        emb_supports = np.zeros([self.batch_size, max(corpus["support_lengths"]), self.emb_matrix.shape[1]])
        emb_questions = np.zeros([self.batch_size, max(corpus["question_lengths"]), self.emb_matrix.shape[1]])

        corpus_ids = deep_map(corpus, self.shared_vocab_config.vocab, ['question', 'support'])

        for i, j in enumerate(len(corpus_ids["question"])):
            for k in range(len(corpus_ids["support"][j])):
                emb_supports[i, k] = self._get_emb(corpus_ids["support"][j][k])
            for k in range(len(corpus_ids["question"][j])):
                emb_supports[i, k] = self._get_emb(corpus_ids["question"][j][k])

        output = {
            FastQAPorts.emb_support: emb_supports,
            FastQAPorts.support_length: corpus["support_lengths"],
            FastQAPorts.emb_question: emb_questions,
            FastQAPorts.question_length: corpus["question_lengths"],
            FastQAPorts.word_in_question: word_in_question,
            FastQAPorts.is_eval: True,
            FastQAPorts.token_char_offsets: token_offsets
        }

        output = numpify(output, [FastQAPorts.word_in_question, FastQAPorts.token_char_offsets])

        return numpify(output)