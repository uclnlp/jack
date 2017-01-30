from jtr.jack.example import ExampleInputModule, ExampleModelModule, ExampleOutputModule
from jtr.jack.tasks.xqa.modules import *


def example_reader(vocab, config):
    """
    Creates an example multiple choice reader.
    Args:
        vocab:
        config:

    Returns:

    """
    resources = SharedVocabAndConfig(vocab, config)
    input_module = ExampleInputModule(resources)
    model_module = ExampleModelModule(resources)
    output_module = ExampleOutputModule()
    reader = JTReader(input_module, model_module, output_module, resources)
    return reader


def fastqa_reader(vocab, config, batch_size=1, dropout=0.0, seed=123):
    """
    Creates a FastQA reader instance (extractive qa model).
    Args:
        vocab:
        config:

    Returns:

    """
    shared_resource = SharedVocabAndConfig(vocab, config)
    return JTReader(XqaWiqInputModule(shared_resource),
                    xqa_wiq_with_min_crossentropy_loss(fastqa_model),
                    XqaOutputModule())


models = globals()