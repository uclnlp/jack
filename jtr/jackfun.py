from jtr.jack import *


def model_module(input_defs: List[TensorPort],
                 output_defs: List[TensorPort]):
    """
    This (meta-)decorator creates a decorator that
    takes functions from input tensors to output tensors and turns them into ModelModules.
    Args:
        input_defs: the input tensor types of the module.
        output_def: the output tensor types of the module.
        loss_def: the loss tensor type

    Returns: a decorator that turns functions into ModelModules.
    """

    def create(f):
        class MyModelModule(SimpleModelModule):

            @property
            def output_ports(self) -> List[TensorPort]:
                return output_defs

            @property
            def input_ports(self) -> List[TensorPort]:
                return input_defs

            def create(self, *tensors: tf.Tensor) -> List[TensorPort]:
                return f(*tensors)

        return MyModelModule()

    return create


def model_module_factory(input_defs: List[TensorPort],
                         output_defs: List[TensorPort],
                         port2ph):
    def create(f):
        return model_module(input_defs, output_defs)(f)

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
