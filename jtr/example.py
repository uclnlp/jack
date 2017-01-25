from jtr.jack import *
import tensorflow as tf


class ExampleInputModule(InputModule):

    def training_generator(self, training_set: List[(Input, Answer)]) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        pass

    def __call__(self, inputs: List[Input]) -> Mapping[TensorPort, np.ndarray]:
        pass

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.multiple_support, Ports.question]


class ExampleModelModule(ModelModule):
    @property
    def output_port(self) -> TensorPort:
        return Ports.scores

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.multiple_support, Ports.question]

    @property
    def loss_port(self) -> TensorPort:
        return Ports.loss

    def create(self, support: tf.Tensor, question: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        return None, None


class ExampleOutputModule(OutputModule):
    @property
    def input_port(self) -> TensorPort:
        return Ports.scores

    def __call__(self, inputs: List[Input], model_results: Mapping[TensorPort, np.ndarray]) -> List[Answer]:
        return []


data_set = [Input(["a is true", "b isn't"], "which is it?", ["a", "b", "c"])]

example_reader = Reader(ExampleInputModule(), ExampleModelModule(), ExampleOutputModule())
example_reader.train(data_set)

answers = example_reader(data_set)
