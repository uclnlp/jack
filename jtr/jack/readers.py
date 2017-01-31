from jtr.jack.example import ExampleInputModule, ExampleModelModule, ExampleOutputModule
from jtr.jack.tasks.xqa.modules import *

models = {}
reader = lambda f: models.setdefault(f.__name__, f)

@reader
def example_reader(vocab, config):
    """ Creates an example multiple choice reader. """
    resources = SharedVocabAndConfig(vocab, config)
    input_module = ExampleInputModule(resources)
    model_module = ExampleModelModule(resources)
    output_module = ExampleOutputModule()
    reader = JTReader(input_module, model_module, output_module, resources)
    return reader

@reader
def fastqa_reader(vocab, config):
    """ Creates a FastQA reader instance (extractive qa model). """
    shared_resource = SharedVocabAndConfig(vocab, config)
    return JTReader(XqaWiqInputModule(shared_resource),
                    xqa_wiq_with_min_crossentropy_loss(fastqa_model),
                    XqaOutputModule())