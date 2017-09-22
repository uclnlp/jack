# -*- coding: utf-8 -*-

from jtr.core import *

from jtr.util.hooks import XQAEvalHook, ClassificationEvalHook

readers = {}
eval_hooks = {}

xqa_readers = {}
genqa_readers = {}
mcqa_readers = {}
kbp_readers = {}


def get_reader_by_name(reader_name) -> JTReader:
    return readers[reader_name]()


def __reader(f):
    readers.setdefault(f.__name__, f)
    return f


def __xqa_reader(f):
    constructor = __reader(f)
    xqa_readers.setdefault(f.__name__, constructor)
    eval_hooks.setdefault(f.__name__, XQAEvalHook)
    return f


def __mcqa_reader(f):
    constructor = __reader(f)
    mcqa_readers.setdefault(f.__name__, constructor)
    eval_hooks.setdefault(f.__name__, ClassificationEvalHook)
    # TODO eval hook
    return f


def __kbp_reader(f):
    from jtr.util.hooks import KBPEvalHook
    constructor = __reader(f)
    kbp_readers.setdefault(f.__name__, constructor)
    eval_hooks.setdefault(f.__name__, KBPEvalHook)
    return f


def __genqa_reader(f):
    constructor = __reader(f)
    genqa_readers.setdefault(f.__name__, constructor)
    # TODO eval hook
    return f


@__mcqa_reader
def example_reader():
    """ Creates an example multiple choice reader. """
    from jtr.tasks.mcqa.simple_mcqa import SimpleMCInputModule, SimpleMCModelModule, SimpleMCOutputModule
    input_module = SimpleMCInputModule()
    model_module = SimpleMCModelModule()
    output_module = SimpleMCOutputModule()
    return JTReader(input_module, model_module, output_module)


@__kbp_reader
def modelf_reader():
    """ Creates a simple kbp reader. """
    from jtr.tasks.kbp.model_f import ModelFInputModule, ModelFModelModule, ModelFOutputModule, KBPReader
    input_module = ModelFInputModule()
    model_module = ModelFModelModule()
    output_module = ModelFOutputModule()
    return KBPReader(input_module, model_module, output_module)


@__kbp_reader
def distmult_reader():
    """ Creates a simple kbp reader. """
    from jtr.tasks.kbp.models import KnowledgeGraphEmbeddingInputModule, KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule, KBPReader
    input_module = KnowledgeGraphEmbeddingInputModule()
    model_module = KnowledgeGraphEmbeddingModelModule(model_name='DistMult')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return KBPReader(input_module, model_module, output_module)


@__kbp_reader
def complex_reader():
    """ Creates a simple kbp reader. """
    from jtr.tasks.kbp.models import KnowledgeGraphEmbeddingInputModule, KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule, KBPReader
    input_module = KnowledgeGraphEmbeddingInputModule()
    model_module = KnowledgeGraphEmbeddingModelModule(model_name='ComplEx')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return KBPReader(input_module, model_module, output_module)


@__kbp_reader
def transe_reader():
    """ Creates a simple kbp reader. """
    from jtr.tasks.kbp.models import KnowledgeGraphEmbeddingInputModule, KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule, KBPReader
    input_module = KnowledgeGraphEmbeddingInputModule()
    model_module = KnowledgeGraphEmbeddingModelModule(model_name='TransE')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return KBPReader(input_module, model_module, output_module)


@__xqa_reader
def fastqa_reader():
    """ Creates a FastQA reader instance (extractive qa model). """
    from jtr.tasks.xqa.fastqa import FastQAInputModule, fastqa_model_module
    from jtr.tasks.xqa.shared import XQAOutputModule

    input_module = FastQAInputModule()
    model_module = fastqa_model_module()
    output_module = XQAOutputModule()
    return JTReader(input_module, model_module, output_module)


@__xqa_reader
def cbow_xqa_reader():
    """ Creates a FastQA reader instance (extractive qa model). """
    from jtr.tasks.xqa.cbow_baseline import CBOWXqaInputModule

    from jtr.tasks.xqa.cbow_baseline import cbow_xqa_model_module
    from jtr.tasks.xqa.shared import XQANoScoreOutputModule

    input_module = CBOWXqaInputModule()
    model_module = cbow_xqa_model_module()
    output_module = XQANoScoreOutputModule()
    return JTReader(input_module, model_module, output_module)


@__mcqa_reader
def cbilstm_snli_reader():
    """
    Creates a SNLI reader instance (multiple choice qa model).
    This particular reader uses a conditional Bidirectional LSTM, as described in [1].

    [1] Tim Rocktäschel et al. - Reasoning about Entailment with Neural Attention. ICLR 2016
    """
    from jtr.tasks.mcqa.simple_mcqa import MultiSupportFixedClassInputs, PairOfBiLSTMOverSupportAndQuestionModel, \
        EmptyOutputModule
    input_module = MultiSupportFixedClassInputs()
    model_module = PairOfBiLSTMOverSupportAndQuestionModel()
    output_module = EmptyOutputModule()
    return JTReader(input_module, model_module, output_module)


@__mcqa_reader
def dam_snli_reader():
    """
    Creates a SNLI reader instance (multiple choice qa model).
    This particular reader uses a Decomposable Attention Model, as described in [1].

    [1] Ankur P. Parikh et al. - A Decomposable Attention Model for Natural Language Inference. EMNLP 2016
    """
    from jtr.tasks.mcqa.simple_mcqa import MultiSupportFixedClassInputs, DecomposableAttentionModel, EmptyOutputModule
    input_module = MultiSupportFixedClassInputs()
    model_module = DecomposableAttentionModel()
    output_module = EmptyOutputModule()
    return JTReader(input_module, model_module, output_module)


@__mcqa_reader
def esim_snli_reader():
    """
    Creates a SNLI reader instance (multiple choice qa model).
    This particular reader uses an Enhanced LSTM Model (ESIM), as described in [1].

    [1] Qian Chen et al. - Enhanced LSTM for Natural Language Inference. ACL 2017
    """
    from jtr.tasks.mcqa.simple_mcqa import MultiSupportFixedClassInputs, ESIMModel, EmptyOutputModule
    input_module = MultiSupportFixedClassInputs()
    model_module = ESIMModel()
    output_module = EmptyOutputModule()
    return JTReader(input_module, model_module, output_module)


@__mcqa_reader
def cbilstm_snli_streaming_reader():
    """
    Creates a SNLI reader instance (multiple choice qa model).
    This particular reader uses a conditional Bidirectional LSTM, as described in [1].

    [1] Tim Rocktäschel et al. - Reasoning about Entailment with Neural Attention. ICLR 2016
    """
    from jtr.tasks.mcqa.simple_mcqa import PairOfBiLSTMOverSupportAndQuestionModel, EmptyOutputModule
    from jtr.tasks.mcqa.streaming_mcqa import StreamingSingleSupportFixedClassInputs
    input_module = StreamingSingleSupportFixedClassInputs()
    model_module = PairOfBiLSTMOverSupportAndQuestionModel()
    output_module = EmptyOutputModule()

    return JTReader(input_module, model_module, output_module)
