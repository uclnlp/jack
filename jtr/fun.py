# -*- coding: utf-8 -*-

from jtr.core import *


def simple_model_module(input_ports: Sequence[TensorPort],
                        output_ports: Sequence[TensorPort],
                        training_input_ports: Sequence[TensorPort],
                        training_output_ports: Sequence[TensorPort]):
    """
    This (meta-)decorator creates a decorator that
    takes functions from input tensors to output tensors and turns them into ModelModules.
    Args:
        input_ports: input ports.
        output_ports: output ports.
        training_input_ports: training input ports.
        training_output_ports: training output ports.

    Returns: a decorator that turns functions into ModelModules.
    """
    def create(shared_vocab_config, f, g):
        class MyModelModule(SimpleModelModule):
            @property
            def output_ports(self) -> Sequence[TensorPort]:
                return output_ports

            @property
            def input_ports(self) -> Sequence[TensorPort]:
                return input_ports

            @property
            def training_input_ports(self) -> Sequence[TensorPort]:
                return training_input_ports

            @property
            def training_output_ports(self) -> Sequence[TensorPort]:
                return training_output_ports

            def create_output(self, shared_resources: SharedResources, *tensors: tf.Tensor) -> Sequence[TensorPort]:
                return f(shared_resources, *tensors)

            def create_training_output(self, shared_resources: SharedResources, *tensors: tf.Tensor)\
                    -> Sequence[TensorPort]:
                return g(shared_resources, *tensors)

        return MyModelModule(shared_vocab_config)

    return create


def no_shared_resources(f):
    def g(_: SharedResources, *tensors: tf.Tensor):
        return f(*tensors)
    return g
