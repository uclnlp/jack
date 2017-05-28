# -*- coding: utf-8 -*-

from jtr.core import *

from jtr.util.hooks import XQAEvalHook, ClassificationEvalHook

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
    __reader(f)
    mcqa_readers.setdefault(f.__name__, f)
    eval_hooks.setdefault(f.__name__, ClassificationEvalHook)
    # TODO eval hook
    return f


def __kbp_reader(f):
    from jtr.util.hooks import KBPEvalHook
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
def example_reader(shared_resources: SharedVocabAndConfig):
    """ Creates an example multiple choice reader. """
    from jtr.tasks.mcqa.simple_mcqa import SimpleMCInputModule, SimpleMCModelModule, SimpleMCOutputModule
    input_module = SimpleMCInputModule(shared_resources)

    model_module = SimpleMCModelModule(shared_resources)

    output_module = SimpleMCOutputModule()

    return JTReader(shared_resources, input_module, model_module, output_module)


@__kbp_reader
def modelf_reader(shared_resources: SharedVocabAndConfig):
    """ Creates a simple kbp reader. """
    from jtr.tasks.kbp.model_f import ModelFInputModule, ModelFModelModule, ModelFOutputModule, KBPReader
    input_module = ModelFInputModule(shared_resources)
    model_module = ModelFModelModule(shared_resources)
    output_module = ModelFOutputModule()
    return KBPReader(shared_resources, input_module, model_module, output_module)


@__kbp_reader
def distmult_reader(shared_resources: SharedVocabAndConfig):
    """ Creates a simple kbp reader. """
    from jtr.tasks.kbp.models import KnowledgeGraphEmbeddingInputModule, KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule, KBPReader
    input_module = KnowledgeGraphEmbeddingInputModule(shared_resources)
    model_module = KnowledgeGraphEmbeddingModelModule(shared_resources, model_name='DistMult')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return KBPReader(shared_resources, input_module, model_module, output_module)


@__kbp_reader
def complex_reader(shared_resources: SharedVocabAndConfig):
    """ Creates a simple kbp reader. """
    from jtr.tasks.kbp.models import KnowledgeGraphEmbeddingInputModule, KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule, KBPReader
    input_module = KnowledgeGraphEmbeddingInputModule(shared_resources)
    model_module = KnowledgeGraphEmbeddingModelModule(shared_resources, model_name='ComplEx')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return KBPReader(shared_resources, input_module, model_module, output_module)


@__kbp_reader
def transe_reader(shared_resources: SharedVocabAndConfig):
    """ Creates a simple kbp reader. """
    from jtr.tasks.kbp.models import KnowledgeGraphEmbeddingInputModule, KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule, KBPReader
    input_module = KnowledgeGraphEmbeddingInputModule(shared_resources)
    model_module = KnowledgeGraphEmbeddingModelModule(shared_resources, model_name='TransE')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return KBPReader(shared_resources, input_module, model_module, output_module)


@__xqa_reader
def fastqa_reader(shared_resources: SharedVocabAndConfig):
    """ Creates a FastQA reader instance (extractive qa model). """
    from jtr.tasks.xqa.fastqa import FastQAInputModule, fatqa_model_module
    from jtr.tasks.xqa.shared import XQAOutputModule

    input_module = FastQAInputModule(shared_resources)

    model_module = fatqa_model_module(shared_resources)

    output_module = XQAOutputModule(shared_resources)

    return JTReader(shared_resources, input_module, model_module, output_module)


@__xqa_reader
def cbow_xqa_reader(shared_resources: SharedVocabAndConfig):
    """ Creates a FastQA reader instance (extractive qa model). """
    from jtr.tasks.xqa.cbow_baseline import CBOWXqaInputModule

    from jtr.tasks.xqa.cbow_baseline import cbow_xqa_model_module
    from jtr.tasks.xqa.shared import XQANoScoreOutputModule

    input_module = CBOWXqaInputModule(shared_resources)

    model_module = cbow_xqa_model_module(shared_resources)

    output_module = XQANoScoreOutputModule(shared_resources)

    return JTReader(shared_resources, input_module, model_module, output_module)


@__mcqa_reader
def snli_reader(shared_resources: SharedVocabAndConfig):
    """ Creates a SNLI reader instance (multiple choice qa model). """
    from jtr.tasks.mcqa.simple_mcqa import SingleSupportFixedClassInputs, PairOfBiLSTMOverSupportAndQuestionModel, \
        EmptyOutputModule

    input_module = SingleSupportFixedClassInputs(shared_resources)

    model_module = PairOfBiLSTMOverSupportAndQuestionModel(shared_resources)

    output_module = EmptyOutputModule()

    return JTReader(shared_resources, input_module, model_module, output_module)


@__mcqa_reader
def dam_snli_reader(shared_resources: SharedVocabAndConfig):
    """ Creates a SNLI reader instance (multiple choice qa model). """
    from jtr.tasks.mcqa.simple_mcqa import SingleSupportFixedClassInputs, DecomposableAttentionModel, EmptyOutputModule
    input_module = SingleSupportFixedClassInputs(shared_resources)
    model_module = DecomposableAttentionModel(shared_resources)
    output_module = EmptyOutputModule()
    return JTReader(shared_resources, input_module, model_module, output_module)
