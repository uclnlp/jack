from jtr.jack import *
from typing import List

def model_module(input_ports: List[TensorPort],
                 output_ports: List[TensorPort],
                 training_input_ports: List[TensorPort],
                 training_ouptut_ports: List[TensorPort]):
    """
    This (meta-)decorator creates a decorator that
    takes functions from input tensors to output tensors and turns them into ModelModules.
    Args:
        input_defs: the input tensor types of the module.
        output_def: the output tensor types of the module.
        loss_def: the loss tensor type

    Returns: a decorator that turns functions into ModelModules.
    """

    def create(f, g):
        class MyModelModule(SimpleModelModule):

            @property
            def output_ports(self) -> List[TensorPort]:
                return output_ports

            @property
            def input_ports(self) -> List[TensorPort]:
                return input_ports

            def training_input_ports(self) -> Mapping[TensorPort, tf.Tensor]:
                return training_input_ports

            def training_output_ports(self) -> List[TensorPort]:
                return training_ouptut_ports

            def create_output(self, *tensors: tf.Tensor) -> List[TensorPort]:
                return f(*tensors)

            def create_training_output(self, *tensors: tf.Tensor) -> List[TensorPort]:
                return g(*tensors)

        return MyModelModule()

    return create


def model_module_factory(input_ports: List[TensorPort],
                         output_ports: List[TensorPort],
                         training_input_ports: List[TensorPort],
                         training_ouptut_ports: List[TensorPort]):
    def create(f, g):
        return model_module(input_ports, training_input_ports, output_ports, training_ouptut_ports)(f, g)

    return create


@model_module([Ports.single_support,
               Ports.question,
               Ports.atomic_candidates], [Ports.candidate_scores])
def average_model_multi_choice(supports: tf.Tensor,
                               question: tf.Tensor,
                               candidates: tf.Tensor) -> List[tf.Tensor]:
    return None, None


@model_module_factory([Ports.single_support,
                       Ports.question,
                       Ports.atomic_candidates], [Ports.candidate_scores, Ports.loss])
def model_multi_choice(pooling_op):
    def model(supports: tf.Tensor,
              question: tf.Tensor,
              candidates: tf.Tensor) -> List[tf.Tensor]:
        return None

    return model
