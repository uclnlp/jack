from jtr.jack import *
import tensorflow as tf

from jtr.jack.data_structures import *
from jtr.pipelines import pipeline, deep_seq_map, deep_map
from jtr.preprocess.batch import get_batches, GeneratorWithRestart
from jtr.preprocess.vocab import Vocab


class ExampleInputModule(InputModule):
    def store(self):
        pass

    def __init__(self, shared_resource):
        self.vocab = shared_resource.vocab
        self.config = shared_resource.config
        self.shared_resource = shared_resource

    def preprocess(self, data, test_time=False):
        corpus = {"support": [], "question": [], "candidates": []}
        if not test_time:
            corpus["answers"] = []
        for xy in data:
            if test_time:
                x = xy
                y = None
            else:
                x, y = xy
            corpus["support"].append(x.support)
            corpus["question"].append(x.question)
            corpus["candidates"].append(x.candidates)
            if not test_time:
                corpus["answers"].append([y.text])
        corpus, _, _, _ = pipeline(corpus, self.vocab, sepvocab=False,
                                   test_time=test_time)
        return corpus

    def setup(self, data: List[Tuple[Question, Answer]]):
        self.preprocess(data)
        return self.shared_resource

    def dataset_generator(self, dataset: List[Tuple[Question, Answer]], is_eval:bool) \
            -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus, _ = self.preprocess(dataset)
        xy_dict = {
            Ports.multiple_support: corpus["support"],
            Ports.question: corpus["question"],
            Ports.atomic_candidates: corpus["candidates"],
            Ports.candidate_targets: corpus["answers"]
        }
        return get_batches(xy_dict)

    def __call__(self, inputs: List[Question]) -> Mapping[TensorPort, np.ndarray]:
        corpus, vocab = self.preprocess(inputs, test_time=True)
        x_dict = {
            Ports.multiple_support: corpus["support"],
            Ports.question: corpus["question"],
            Ports.atomic_candidates: corpus["candidates"]
        }

        # todo

        return x_dict

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.multiple_support, Ports.question, Ports.atomic_candidates, Ports.candidate_targets]


class ExampleModelModule(SimpleModelModule):

    def __init__(self, vocab: SharedVocab, config=None):
        self.vocab = vocab.vocab
        self.embeddings = None
        super().__init__()

    @property
    def output_ports(self) -> TensorPort:
        return [Ports.scores, Ports.loss]

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.multiple_support, Ports.question, Ports.atomic_candidates, Ports.candidate_targets]

    # output scores and loss tensor
    def create_output(self, support: tf.Tensor, question: tf.Tensor,
                      candidates: tf.Tensor, target: tf.Tensor) -> Mapping[TensorPort, tf.Tensor]:
        input_size = 10
        self.embeddings = tf.get_variable(
            "embeddings", [len(self.vocab), input_size],
            trainable=True, dtype="float32")

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

        # todo

        # tf.nn.embedding_lookup(self.embeddings, xs)

        scores = 0.0
        loss = tf.Variable(0.0)

        return {
            Ports.scores: scores,
            Ports.loss: loss
        }


class ExampleOutputModule(OutputModule):
    def store(self):
        pass

    @property
    def input_ports(self) -> TensorPort:
        return [Ports.scores]

    def __call__(self, inputs: List[Question], model_results: Mapping[TensorPort, np.ndarray]) -> List[Answer]:
        return []


if __name__ == '__main__':
    data_set = [
        (Question(["a is true", "b isn't"], "which is it?", ["a", "b", "c"]),
         Answer("a", 1.0))
    ]

    vocab = SharedVocabAndConfig(Vocab())
    example_reader = JTReader(ExampleInputModule(vocab),
                              ExampleModelModule(vocab),
                              ExampleOutputModule(),
                              vocab)

    example_reader.setup(data_set)

    # todo: chose optimizer based on config
    example_reader.train(data_set, optim=tf.train.AdamOptimizer())

    # answers = example_reader(data_set)
