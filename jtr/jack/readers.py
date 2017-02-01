from jtr.jack import *

readers = {}

xqa_readers = {}
genqa_readers = {}
mcqa_readers = {}


def __reader(f):
    readers.setdefault(f.__name__, f)
    return f


def __xqa_reader(f):
    __reader(f)
    xqa_readers.setdefault(f.__name__, f)
    return f


def __mcqa_reader(f):
    __reader(f)
    mcqa_readers.setdefault(f.__name__, f)
    return f


def __genqa_reader(f):
    __reader(f)
    genqa_readers.setdefault(f.__name__, f)
    return f


@__mcqa_reader
def example_reader(vocab, config):
    """ Creates an example multiple choice reader. """
    from jtr.jack.example import ExampleInputModule, ExampleModelModule, ExampleOutputModule
    resources = SharedVocabAndConfig(vocab, config)
    input_module = ExampleInputModule(resources)
    model_module = ExampleModelModule(resources)
    output_module = ExampleOutputModule()
    jtreader = JTReader(input_module, model_module, output_module, resources)
    return jtreader


@__xqa_reader
def fastqa_reader(vocab, config):
    """ Creates a FastQA reader instance (extractive qa model). """
    from jtr.jack.tasks.xqa.fastqa import FastQAInputModule
    from jtr.jack.tasks.xqa.fastqa import fastqa_with_min_crossentropy_loss
    from jtr.jack.tasks.xqa.shared import XqaOutputModule
    from jtr.jack.tf_fun.fastqa import fastqa_model

    shared_resource = SharedVocabAndConfig(vocab, config)
    return JTReader(FastQAInputModule(shared_resource),
                    fastqa_with_min_crossentropy_loss(fastqa_model),
                    XqaOutputModule())