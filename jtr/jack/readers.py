from jtr.jack.example import ExampleInputModule, ExampleModelModule, ExampleOutputModule
from jtr.jack.tasks.xqa.modules import *

models = {}


def reader(f):
    models.setdefault(f.__name__, f)
    return f


@reader
def example_reader(vocab, config):
    """ Creates an example multiple choice reader. """
    resources = SharedVocabAndConfig(vocab, config)
    input_module = ExampleInputModule(resources)
    model_module = ExampleModelModule(resources)
    output_module = ExampleOutputModule()
    jtreader = JTReader(input_module, model_module, output_module, resources)
    return jtreader


@reader
def fastqa_reader(vocab, config):
    """ Creates a FastQA reader instance (extractive qa model). """
    shared_resource = SharedVocabAndConfig(vocab, config)
    return JTReader(XqaWiqInputModule(shared_resource),
                    xqa_wiq_with_min_crossentropy_loss(fastqa_model),
                    XqaOutputModule())