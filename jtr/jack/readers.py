from jtr.jack import *

readers = {}
eval_hooks = {}

xqa_readers = {}
genqa_readers = {}
mcqa_readers = {}


def __reader(f):
    readers.setdefault(f.__name__, f)
    return f


def __xqa_reader(f):
    from jtr.jack.train.hooks import XQAEvalHook
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
    shared_resources = SharedVocabAndConfig(vocab, config)
    input_module = SimpleMCInputModule(shared_resources)
    model_module = SimpleMCModelModule(shared_resources)
    output_module = SimpleMCOutputModule()
    jtreader = JTReader(shared_resources, input_module, model_module, output_module)
    return jtreader


@__xqa_reader
def fastqa_reader(vocab, config):
    """ Creates a FastQA reader instance (extractive qa model). """
    from jtr.jack.tasks.xqa.fastqa import FastQAInputModule, fatqa_model_module
    from jtr.jack.tasks.xqa.shared import XQAOutputModule

    shared_resources = SharedVocabAndConfig(vocab, config)
    return JTReader(shared_resources,
                    FastQAInputModule(shared_resources),
                    fatqa_model_module(shared_resources),
                    XQAOutputModule(shared_resources))
