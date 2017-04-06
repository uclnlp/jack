# -*- coding: utf-8 -*-

from jtr.jack.core import *
from jtr.jack.train.hooks import XQAEvalHook, ClassificationEvalHook

readers = {}
eval_hooks = {}

xqa_readers = {}
genqa_readers = {}
mcqa_readers = {}
kbp_readers = {}


def __reader(f):
    readers.setdefault(f.__name__, f)
    return f


def __xqa_reader(f):
    __reader(f)
    xqa_readers.setdefault(f.__name__, f)
    eval_hooks.setdefault(f.__name__, XQAEvalHook)
    return f


def __mcqa_reader(f):
    from jtr.jack.train.hooks import XQAEvalHook
    __reader(f)
    mcqa_readers.setdefault(f.__name__, f)
    eval_hooks.setdefault(f.__name__, ClassificationEvalHook)
    # TODO eval hook
    return f


def __kbp_reader(f):
    from jtr.jack.train.hooks import KBPEvalHook
    __reader(f)
    kbp_readers.setdefault(f.__name__, f)
    eval_hooks.setdefault(f.__name__, KBPEvalHook)
    return f


def __genqa_reader(f):
    __reader(f)
    genqa_readers.setdefault(f.__name__, f)
    # TODO eval hook
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


@__kbp_reader
def modelf_reader(vocab, config):
    """ Creates a simple kbp reader. """
    from jtr.jack.tasks.kbp.model_f import ModelFInputModule, ModelFModelModule, ModelFOutputModule, KBPReader
    shared_resources = SharedVocabAndConfig(vocab, config)
    input_module = ModelFInputModule(shared_resources)
    model_module = ModelFModelModule(shared_resources)
    output_module = ModelFOutputModule()
    jtreader = KBPReader(shared_resources, input_module, model_module, output_module)
    return jtreader



@__xqa_reader
def fastqa_reader(vocab, config=dict()):
    """ Creates a FastQA reader instance (extractive qa model). """
    from jtr.jack.tasks.xqa.fastqa import FastQAInputModule, fatqa_model_module
    from jtr.jack.tasks.xqa.shared import XQAOutputModule

    shared_resources = SharedVocabAndConfig(vocab, config)
    return JTReader(shared_resources,
                    FastQAInputModule(shared_resources),
                    fatqa_model_module(shared_resources),
                    XQAOutputModule(shared_resources))


@__xqa_reader
def cbow_xqa_reader(vocab, config=dict()):
    """ Creates a FastQA reader instance (extractive qa model). """
    from jtr.jack.tasks.xqa.cbow_baseline import cbow_xqa_model_module
    from jtr.jack.tasks.xqa.shared import XQANoScoreOutputModule
    from jtr.jack.tasks.xqa.cbow_baseline import CBOWXqaInputModule

    shared_resources = SharedVocabAndConfig(vocab, config)
    return JTReader(shared_resources,
                    CBOWXqaInputModule(shared_resources),
                    cbow_xqa_model_module(shared_resources),
                    XQANoScoreOutputModule(shared_resources))


@__mcqa_reader
def snli_reader(vocab, config):
    """ Creates a SNLI reader instance (multiple choice qa model). """
    from jtr.jack.tasks.mcqa.simple_mcqa import SingleSupportFixedClassInputs, PairOfBiLSTMOverSupportAndQuestionModel, EmptyOutputModule
    shared_resources = SharedVocabAndConfig(vocab, config)
    return JTReader(shared_resources,
                    SingleSupportFixedClassInputs(shared_resources),
                    PairOfBiLSTMOverSupportAndQuestionModel(shared_resources),
                    EmptyOutputModule())
