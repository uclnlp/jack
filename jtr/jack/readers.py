from jtr.jack import *
from jtr.jack.train.hooks import XQAEvalHook

readers = {}
eval_hooks = {}

xqa_readers = {}
genqa_readers = {}
mcqa_readers = {}


def __reader(f):
    readers.setdefault(f.__name__, f)
    return f


def __xqa_reader(f):
    __reader(f)
    xqa_readers.setdefault(f.__name__, f)
    eval_hooks.setdefault(f.__name__, XQAEvalHook)
    return f


def __mcqa_reader(f):
    __reader(f)
    mcqa_readers.setdefault(f.__name__, f)
    #TODO eval hook
    return f


def __genqa_reader(f):
    __reader(f)
    genqa_readers.setdefault(f.__name__, f)
    #TODO eval hook
    return f


@__mcqa_reader
def example_reader(vocab, config):
    """ Creates an example multiple choice reader. """
    from jtr.jack.tasks.mcqa.simple_mcqa import SimpleMCInputModule, SimpleMCModelModule, SimpleMCOutputModule
    resources = SharedVocabAndConfig(vocab, config)
    input_module = SimpleMCInputModule(resources)
    model_module = SimpleMCModelModule(resources)
    output_module = SimpleMCOutputModule()
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