from jtr.jack import *
import tensorflow as tf
from jtr.pipelines import pipeline
from typing import Mapping
from jtr.preprocess.batch import get_batches


class ExampleInputModule(InputModule):
    def __init__(self, vocab, config):
        pass

    def training_generator(self, training_set: List[Tuple[Input, Answer]]) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus = {"support": [], "question": [], "candidates": []}
        # fixme: not sure how to use answer here
        for input, answer in training_set:
            corpus["support"].append(input.support)
            corpus["question"].append(input.question)
            corpus["candidates"].append(input.candidates)
        # todo: I have the feeling we can't easily decouple input from model
        # module as the model needs access to the vocab (likewise the output)
        corpus, vocab, target_vocab, candidate_vocab = pipeline(corpus)
        output = {
            Ports.multiple_support: corpus["support"],
            Ports.question: corpus["question"],
            Ports.atomic_candidates: corpus["candidates"],
        }

        return get_batches(output)

    def __call__(self, inputs: List[Input]) -> Mapping[TensorPort, np.ndarray]:
        corpus = {"support": [], "question": [], "candidates": []}
        for input in inputs:
            corpus["support"].append(input.support)
            corpus["question"].append(input.question)
            corpus["candidates"].append(input.candidates)
        corpus, vocab, target_vocab, candidate_vocab = pipeline(corpus, test_time=True)
        output = {
            Ports.multiple_support: corpus["support"],
            Ports.question: corpus["question"],
            Ports.atomic_candidates: corpus["candidates"]
        }
        return output

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.multiple_support, Ports.question]

    @property
    def target_port(self):
        return Ports.candidate_targets


class ExampleModelModule(SimpleModelModule):
    def __init__(self, vocab, config):
        pass

    @property
    def target_port(self) -> TensorPort:
        return Ports.candidate_targets

    @property
    def output_port(self) -> TensorPort:
        return Ports.scores

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.multiple_support, Ports.question]

    @property
    def loss_port(self) -> TensorPort:
        return Ports.loss

    # output scores and loss tensor
    def create(self, target: tf.Tensor, support: tf.Tensor, question: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        with tf.variable_scope("embedders") as varscope:
            question_embedded = question  # nvocab(question)
            varscope.reuse_variables()
            # fixme: where does this model module get its candidates from?
            candidates_embedded = candidates  # nvocab(candidates)

        question_encoding = tf.reduce_sum(question_embedded, 1)

        scores = logits = tf.reduce_sum(
            tf.expand_dims(question_encoding, 1) * candidates_embedded, 2)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(scores, targets),
            name='predictor_loss')

        return scores, loss


class ExampleOutputModule(OutputModule):
    @property
    def input_port(self) -> TensorPort:
        return Ports.scores

    def __call__(self, inputs: List[Input], model_results: Mapping[TensorPort, np.ndarray]) -> List[Answer]:
        return []


data_set = [Input(["a is true", "b isn't"], "which is it?", ["a", "b", "c"])]

example_reader = Reader(ExampleInputModule(), ExampleModelModule(), ExampleOutputModule())
# example_reader.train(data_set)

answers = example_reader(data_set)
