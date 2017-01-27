import random

from jtr.jack import *
from jtr.preprocess.batch import get_batches, GeneratorWithRestart
from jtr.preprocess.map import deep_map, numpify


class FastQAPorts:
    """
    It is good practice define all ports needed for a single model jointly, to get an overview
    """
    question = Ports.question

    support = Ports.single_support

    # This feature is model specific and thus, not part of the conventional Ports
    word_in_question = TensorPort(tf.float32, [None, None], "word_in_question_feature",
                                  "Represents a 1/0 feature for all context tokens denoting"
                                  " whether it is part of the question or not",
                                  "[Q, context_length]")

    # ports used during training
    answer_to_question = Ports.Flat.answer_to_question
    answer_span = Ports.Flat.answer_span


class FastQAInputModule(InputModule):

    def __init__(self, nvocab, batch_size, seed=123):
        self.nvocab = nvocab
        self.batch_size = batch_size
        self._rng = random.Random(seed)

    def output_ports(self) -> List[TensorPort]:
        return [FastQAPorts.question, FastQAPorts.support, FastQAPorts.word_in_question]

    def training_generator(self, training_set: List[Tuple[Input, List[Answer]]]) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus = {"support": [], "question": []}

        answer_spans = []
        for input, answers in training_set:
            corpus["support"].append(input.support)
            corpus["question"].append(input.question)
            answer_spans.append([a.span for a in answers])

        word_in_question = []
        for q, s in zip(corpus["question"], corpus["support"]):
            wiq = []
            for token in s:
                wiq.append(float(token in q))
            word_in_question.append(wiq)

        corpus_ids = deep_map(corpus, self.nvocab, ['question', 'support'])

        def batch_generator():
            todo = self._rng.shuffle(list(range(len(corpus_ids["question"]))))
            while todo:
                supports = list()
                questions = list()
                wiq = list()
                spans = list()
                span2question = []

                for i, j in enumerate(todo[:self.batch_size]):
                    supports.append(corpus_ids["support"][j])
                    questions.append(corpus_ids["question"][j])
                    spans.extend(spans[j])
                    span2question.extend(i for _ in spans[j])
                    wiq.append(word_in_question[j])


                output = {
                    FastQAPorts.support: corpus_ids["support"],
                    FastQAPorts.question: corpus_ids["question"],
                    FastQAPorts.word_in_question: wiq,
                    FastQAPorts.answer_span: spans,
                    FastQAPorts.answer_to_question: span2question
                }
                batch = numpify(output, keys=[FastQAPorts.support, FastQAPorts.question, FastQAPorts.word_in_question])
                yield batch

        return GeneratorWithRestart(batch_generator)

    def __call__(self, inputs: List[Input]) -> Mapping[TensorPort, np.ndarray]:
        corpus = {"support": [], "question": []}
        for input in inputs:
            corpus["support"].append(input.support)
            corpus["question"].append(input.question)

        corpus_ids = deep_map(corpus, self.nvocab, ['question', 'support'])

        word_in_question = []

        for q, s in zip(corpus["question"], corpus["support"]):
            wiq = []
            for token in s:
                wiq.append(float(token in q))
            word_in_question.append(wiq)

        output = {
            FastQAPorts.support: corpus_ids["support"],
            FastQAPorts.question: corpus_ids["question"],
            FastQAPorts.word_in_question: word_in_question
        }

        return numpify(output)
