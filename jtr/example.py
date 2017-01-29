from jtr.jack import *
import tensorflow as tf
from jtr.pipelines import pipeline, deep_seq_map, deep_map
from typing import Mapping
from jtr.preprocess.batch import get_batches


class ExampleInputModule(InputModule):
    def store(self):
        pass

    def __init__(self, vocab=None, config=None):
        self.vocab = vocab
        self.config = config

    def training_generator(self, training_set: List[Tuple[Input, Answer]]) \
            -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus = {"support": [], "question": [], "candidates": [], "answers": []}
        for x, y in training_set:
            corpus["support"].append(x.support)
            corpus["question"].append(x.question)
            corpus["candidates"].append(x.candidates)
            corpus["answers"].append(y.text)

        # todo: for now using the same vocab
        corpus, vocab, _, _ = \
            pipeline(corpus, self.vocab, sepvocab=False)

        if self.vocab is None:
            self.vocab = vocab

        xy_dict = {
            Ports.multiple_support: corpus["support"],
            Ports.question: corpus["question"],
            Ports.atomic_candidates: corpus["candidates"],
            Ports.candidate_targets: corpus["answers"]
        }

        return self.embed(get_batches(xy_dict))

    def embed(self, generator):
        # fixme: hard coded input size should be replaced by config
        input_size = 10
        embeddings = tf.get_variable(
            "embeddings", [len(self.vocab), input_size],
            trainable=True, dtype="float32")

        keys = self.output_ports + [self.target_port]

        def embed_batch(batch):
            return deep_map(
                batch, lambda xs: tf.nn.embedding_lookup(embeddings, xs), keys)

        generator = [embed_batch(x) for x in generator.iterator()]

        return generator

    def __call__(self, inputs: List[Input]) -> Mapping[TensorPort, np.ndarray]:
        corpus = {"support": [], "question": [], "candidates": []}
        for x in inputs:
            corpus["support"].append(x.support)
            corpus["question"].append(x.question)
            corpus["candidates"].append(x.candidates)
        corpus, vocab, target_vocab, candidate_vocab = \
            pipeline(corpus, test_time=True)
        x_dict = {
            Ports.multiple_support: corpus["support"],
            Ports.question: corpus["question"],
            Ports.atomic_candidates: corpus["candidates"]
        }

        # todo

        return x_dict

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.multiple_support, Ports.question, Ports.atomic_candidates]

    @property
    def target_port(self):
        return Ports.candidate_targets


class ExampleModelModule(SimpleModelModule):
    def store(self):
        pass

    def __init__(self, vocab=None, config=None):
        self.vocab = vocab
        super().__init__()

    @property
    def target_port(self) -> TensorPort:
        return Ports.candidate_targets

    @property
    def output_port(self) -> TensorPort:
        return Ports.scores

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.multiple_support, Ports.question, Ports.atomic_candidates]

    @property
    def loss_port(self) -> TensorPort:
        return Ports.loss

    # output scores and loss tensor
    def create(self, target: tf.Tensor, support: tf.Tensor, question: tf.Tensor,
               candidates: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        # with tf.variable_scope("embedders") as varscope:
        #     question_embedded = question  # todo: nvocab(question)
        #     varscope.reuse_variables()
        #     candidates_embedded = candidates  # todo: nvocab(candidates)
        #
        # question_encoding = tf.reduce_sum(question_embedded, 1)
        #
        # scores = tf.reduce_sum(
        #     tf.expand_dims(question_encoding, 1) * candidates_embedded, 2)
        #
        # loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(scores, target),
        #     name='predictor_loss')

        scores = 0
        loss = 0

        return scores, loss


class ExampleOutputModule(OutputModule):
    def store(self):
        pass

    @property
    def input_port(self) -> TensorPort:
        return Ports.scores

    def __call__(self, inputs: List[Input], model_results: Mapping[TensorPort, np.ndarray]) -> List[Answer]:
        return []


data_set = [
    (Input(["a is true", "b isn't"], "which is it?", ["a", "b", "c"]),
     Answer("a", 1.0))
]

example_reader = Reader(ExampleInputModule(),
                        ExampleModelModule(),
                        ExampleOutputModule())

example_reader.train(data_set)

# answers = example_reader(data_set)
