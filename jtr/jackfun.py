from jtr.jack import *


def model_module(input_defs: List[TensorPort],
                 output_def: TensorPort,
                 loss_def: TensorPort):
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
        class MyModelModule(ModelModule):
            def __init__(self):
                super().__init__()

            @property
            def output_def(self) -> TensorPort:
                return output_def

            @property
            def input_defs(self) -> List[TensorPort]:
                return input_defs

            @property
            def loss_def(self) -> TensorPort:
                return loss_def

            def create(self, *input_tensors: tf.Tensor) -> (tf.Tensor, tf.Tensor):
                return f(*input_tensors)

            def __call__(self, *args, **kwargs):
                return f(*args, **kwargs)

        return MyModelModule()

    return create


def model_module_factory(input_defs: List[TensorPort],
                         output_def: TensorPort,
                         loss_def: TensorPort):
    def create(f):
        return model_module(input_defs, output_def, loss_def)(f)

    return create


@model_module([InputPorts.multiple_support,
               InputPorts.question,
               InputPorts.atomic_candidates], OutputPorts.scores, OutputPorts.loss)
def average_model_multi_choice(supports: tf.Tensor,
                               question: tf.Tensor,
                               candidates: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    return None, None


@model_module_factory([InputPorts.multiple_support,
                       InputPorts.question,
                       InputPorts.atomic_candidates], OutputPorts.scores, OutputPorts.loss)
def model_multi_choice(pooling_op):
    def model(supports: tf.Tensor,
              question: tf.Tensor,
              candidates: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        return None

    return model
