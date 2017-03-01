from typing import List

from jtr.jack.core import *


def simple_model_module(input_ports: List[TensorPort],
                        output_ports: List[TensorPort],
                        training_input_ports: List[TensorPort],
                        training_output_ports: List[TensorPort]):
    """
    This (meta-)decorator creates a decorator that
    takes functions from input tensors to output tensors and turns them into ModelModules.
    Args:
        input_defs: the input tensor types of the module.
        output_def: the output tensor types of the module.
        loss_def: the loss tensor type

    Returns: a decorator that turns functions into ModelModules.
    """

    def create(shared_vocab_config, f, g):
        class MyModelModule(SimpleModelModule):
            @property
            def output_ports(self) -> List[TensorPort]:
                return output_ports

            @property
            def input_ports(self) -> List[TensorPort]:
                return input_ports

            @property
            def training_input_ports(self) -> List[TensorPort]:
                return training_input_ports

            @property
            def training_output_ports(self) -> List[TensorPort]:
                return training_output_ports


            def create_output(self, shared_resources: SharedResources, *tensors: tf.Tensor) -> List[TensorPort]:
                return f(shared_resources, *tensors)

            def create_training_output(self, shared_resources: SharedResources, *tensors: tf.Tensor) -> List[
                TensorPort]:
                return g(shared_resources, *tensors)

        return MyModelModule(shared_vocab_config)

    return create


def no_shared_resources(f):
    def g(shared_resources: SharedResources, *tensors: tf.Tensor):
        return f(*tensors)

    return g
