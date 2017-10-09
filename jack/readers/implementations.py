from typing import Union

from jack.core.reader import TFReader
from jack.core.shared_resources import SharedResources
from jack.util.hooks import XQAEvalHook, ClassificationEvalHook

readers = {}
eval_hooks = {}

extractive_qa_readers = {}
nli_readers = {}
kbp_readers = {}


def __reader(f):
    readers.setdefault(f.__name__, f)
    return f


def __extractive_qa_reader(f):
    __reader(f)
    extractive_qa_readers.setdefault(f.__name__, f)
    eval_hooks.setdefault(f.__name__, XQAEvalHook)
    return f


def __nli_reader(f):
    __reader(f)
    nli_readers.setdefault(f.__name__, f)
    eval_hooks.setdefault(f.__name__, ClassificationEvalHook)
    return f


def __kbp_reader(f):
    from jack.util.hooks import KBPEvalHook
    __reader(f)
    kbp_readers.setdefault(f.__name__, f)
    eval_hooks.setdefault(f.__name__, KBPEvalHook)
    return f


def create_shared_resources(resources_or_config: Union[dict, SharedResources] = None) -> SharedResources:
    """
    Produces a SharedResources object based on the input.
    Args:
        resources_or_config: either nothing, a configuration dictionary, or shared resources

    Returns: a SharedResources object.
    """
    if resources_or_config is None:
        return SharedResources()
    elif isinstance(resources_or_config, SharedResources):
        return resources_or_config
    else:
        return SharedResources(config=resources_or_config)


@__nli_reader
def example_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """ Creates an example multiple choice reader. """
    from jack.readers.multiple_choice.shared import SimpleMCInputModule, SimpleMCModelModule, SimpleMCOutputModule
    shared_resources = create_shared_resources(resources_or_conf)
    input_module = SimpleMCInputModule(shared_resources)
    model_module = SimpleMCModelModule(shared_resources)
    output_module = SimpleMCOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)


@__kbp_reader
def modelf_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """ Creates a kbp model F. """
    from jack.readers.kbp.model_f import ModelFInputModule, ModelFModelModule, ModelFOutputModule
    shared_resources = create_shared_resources(resources_or_conf)
    input_module = ModelFInputModule(shared_resources)
    model_module = ModelFModelModule(shared_resources)
    output_module = ModelFOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)


@__kbp_reader
def distmult_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """ Creates a kbp DistMult model. """
    from jack.readers.kbp.models import KnowledgeGraphEmbeddingInputModule, KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule
    shared_resources = create_shared_resources(resources_or_conf)
    input_module = KnowledgeGraphEmbeddingInputModule(shared_resources)
    model_module = KnowledgeGraphEmbeddingModelModule(shared_resources, model_name='DistMult')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)


@__kbp_reader
def complex_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """ Creates a kbp Complex model. """
    from jack.readers.kbp.models import KnowledgeGraphEmbeddingInputModule, KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule
    shared_resources = create_shared_resources(resources_or_conf)
    input_module = KnowledgeGraphEmbeddingInputModule(shared_resources)
    model_module = KnowledgeGraphEmbeddingModelModule(shared_resources, model_name='ComplEx')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)


@__kbp_reader
def transe_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """ Creates a kbp TransE model. """
    from jack.readers.kbp.models import KnowledgeGraphEmbeddingInputModule, KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule
    shared_resources = create_shared_resources(resources_or_conf)

    input_module = KnowledgeGraphEmbeddingInputModule(shared_resources)
    model_module = KnowledgeGraphEmbeddingModelModule(shared_resources, model_name='TransE')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)


@__extractive_qa_reader
def fastqa_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """ Creates a FastQA reader instance (extractive qa model). """
    from jack.readers.extractive_qa.fastqa import FastQAInputModule, fatqa_model_module
    from jack.readers.extractive_qa.shared import XQAOutputModule
    shared_resources = create_shared_resources(resources_or_conf)

    input_module = FastQAInputModule(shared_resources)
    model_module = fatqa_model_module(shared_resources)
    output_module = XQAOutputModule(shared_resources)
    return TFReader(shared_resources, input_module, model_module, output_module)


@__extractive_qa_reader
def cbow_xqa_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """Creates a FastQA reader instance (extractive qa model). """
    from jack.readers.extractive_qa.cbow import CBOWXqaInputModule
    from jack.readers.extractive_qa.cbow import cbow_xqa_model_module
    from jack.readers.extractive_qa.shared import XQANoScoreOutputModule
    shared_resources = create_shared_resources(resources_or_conf)

    input_module = CBOWXqaInputModule(shared_resources)
    model_module = cbow_xqa_model_module(shared_resources)
    output_module = XQANoScoreOutputModule(shared_resources)
    return TFReader(shared_resources, input_module, model_module, output_module)


@__nli_reader
def cbilstm_snli_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """
    Creates a SNLI reader instance (multiple choice qa model).
    This particular reader uses a conditional Bidirectional LSTM, as described in [1].

    [1] Tim Rocktäschel et al. - Reasoning about Entailment with Neural Attention. ICLR 2016
    """
    from jack.readers.multiple_choice.shared import MultiSupportFixedClassInputs
    from jack.readers.natural_language_inference.bilstm import PairOfBiLSTMOverSupportAndQuestionModel
    from jack.readers.multiple_choice.shared import SimpleMCOutputModule
    shared_resources = create_shared_resources(resources_or_conf)

    input_module = MultiSupportFixedClassInputs(shared_resources)
    model_module = PairOfBiLSTMOverSupportAndQuestionModel(shared_resources)
    output_module = SimpleMCOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)


@__nli_reader
def dam_snli_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """
    Creates a SNLI reader instance (multiple choice qa model).
    This particular reader uses a Decomposable Attention Model, as described in [1].

    [1] Ankur P. Parikh et al. - A Decomposable Attention Model for Natural Language Inference. EMNLP 2016
    """
    from jack.readers.multiple_choice.shared import MultiSupportFixedClassInputs
    from jack.readers.natural_language_inference.decomposable_attention import DecomposableAttentionModel
    from jack.readers.multiple_choice.shared import SimpleMCOutputModule
    shared_resources = create_shared_resources(resources_or_conf)

    input_module = MultiSupportFixedClassInputs(shared_resources)
    model_module = DecomposableAttentionModel(shared_resources)
    output_module = SimpleMCOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)


@__nli_reader
def esim_snli_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """
    Creates a SNLI reader instance (multiple choice qa model).
    This particular reader uses an Enhanced LSTM Model (ESIM), as described in [1].

    [1] Qian Chen et al. - Enhanced LSTM for Natural Language Inference. ACL 2017
    """
    from jack.readers.multiple_choice.shared import MultiSupportFixedClassInputs
    from jack.readers.natural_language_inference.esim import ESIMModel
    from jack.readers.multiple_choice.shared import SimpleMCOutputModule
    shared_resources = create_shared_resources(resources_or_conf)

    input_module = MultiSupportFixedClassInputs(shared_resources)
    model_module = ESIMModel(shared_resources)
    output_module = SimpleMCOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)
