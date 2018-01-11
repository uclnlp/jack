import os
from typing import Union

from jack.core.shared_resources import SharedResources
from jack.core.tensorflow import TFReader
from jack.util.hooks import XQAEvalHook, ClassificationEvalHook

readers = {}
eval_hooks = {}

extractive_qa_readers = {}
classification_readers = {}
nli_readers = {}
link_prediction_readers = {}


def __reader(f):
    readers.setdefault(f.__name__, f)
    return f


def extractive_qa_reader(f):
    __reader(f)
    extractive_qa_readers.setdefault(f.__name__, f)
    eval_hooks.setdefault(f.__name__, XQAEvalHook)
    return f


def classification_reader(f):
    __reader(f)
    classification_readers.setdefault(f.__name__, f)
    eval_hooks.setdefault(f.__name__, ClassificationEvalHook)
    return f


def nli_reader(f):
    nli_readers.setdefault(f.__name__, f)
    return classification_reader(f)


def link_prediction_reader(f):
    from jack.util.hooks import LogProbEvalHook
    __reader(f)
    link_prediction_readers.setdefault(f.__name__, f)
    eval_hooks.setdefault(f.__name__, LogProbEvalHook)
    return f


def reader_from_file(load_dir: str, **kwargs):
    """Loads a reader from a file.

    Args:
        load_dir: directory containing the reader being loaded.
        kwargs: can be used to overwrite the reader configuration if necessary.

    Returns:
        the reader.
    """
    shared_resources = create_shared_resources()
    shared_resources.load(os.path.join(load_dir, "shared_resources"))
    if kwargs:
        shared_resources.config.update(kwargs)
    reader = readers[shared_resources.config["reader"]](shared_resources)
    reader.load_and_setup_modules(load_dir)
    return reader


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


# Question Answering


def _tf_extractive_qa_reader(model_module_constructor, resources_or_conf: Union[dict, SharedResources]):
    from jack.readers.extractive_qa.shared import XQAInputModule, XQAOutputModule
    shared_resources = create_shared_resources(resources_or_conf)
    input_module = XQAInputModule(shared_resources)
    model_module = model_module_constructor(shared_resources)
    output_module = XQAOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)


@extractive_qa_reader
def fastqa_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """Creates a FastQA reader instance (extractive qa model)."""
    from jack.readers.extractive_qa.tensorflow.fastqa import FastQAModule
    return _tf_extractive_qa_reader(FastQAModule, resources_or_conf)


@extractive_qa_reader
def modular_qa_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """Creates a FastQA reader instance (extractive qa model)."""
    from jack.readers.extractive_qa.tensorflow.modular_qa_model import ModularQAModel
    return _tf_extractive_qa_reader(ModularQAModel, resources_or_conf)


@extractive_qa_reader
def fastqa_reader_torch(resources_or_conf: Union[dict, SharedResources] = None):
    """ Creates a FastQA reader instance (extractive qa model). """
    from jack.readers.extractive_qa.torch.fastqa import FastQAPyTorchModelModule
    from jack.readers.extractive_qa.shared import XQAInputModule, XQAOutputModule
    from jack.core.torch import PyTorchReader
    shared_resources = create_shared_resources(resources_or_conf)
    input_module = XQAInputModule(shared_resources)
    model_module = FastQAPyTorchModelModule(shared_resources)
    output_module = XQAOutputModule()
    return PyTorchReader(shared_resources, input_module, model_module, output_module)


# Natural Language Inference


def _tf_nli_reader(model_module_constructor, resources_or_conf: Union[dict, SharedResources] = None):
    from jack.readers.classification.shared import ClassificationSingleSupportInputModule
    from jack.readers.classification.shared import SimpleClassificationOutputModule
    shared_resources = create_shared_resources(resources_or_conf)
    input_module = ClassificationSingleSupportInputModule(shared_resources)
    model_module = model_module_constructor(shared_resources)
    output_module = SimpleClassificationOutputModule(shared_resources)
    return TFReader(shared_resources, input_module, model_module, output_module)


@nli_reader
def dam_snli_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """Creates a NLI reader instance.
    This particular reader uses a Decomposable Attention Model, as described in [1].
    [1] Ankur P. Parikh et al. - A Decomposable Attention Model for Natural Language Inference. EMNLP 2016
    """
    from jack.readers.natural_language_inference.decomposable_attention import DecomposableAttentionModel
    return _tf_nli_reader(DecomposableAttentionModel, resources_or_conf)


@nli_reader
def cbilstm_nli_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """Creates a NLI reader based on conditional BiLSTMs."""
    from jack.readers.natural_language_inference.conditional_bilstm import ConditionalBiLSTMClassificationModel
    return _tf_nli_reader(ConditionalBiLSTMClassificationModel, resources_or_conf)


@nli_reader
def modular_nli_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """Creates a Modular NLI reader instance. Model defined in config."""
    from jack.readers.natural_language_inference.modular_nli_model import ModularNLIModel
    return _tf_nli_reader(ModularNLIModel, resources_or_conf)


# Link Prediction/Knowledge Base Population Models


@link_prediction_reader
def distmult_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """Creates a knowledge_base_population DistMult model."""
    from jack.readers.link_prediction.models import KnowledgeGraphEmbeddingInputModule, \
        KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule
    shared_resources = create_shared_resources(resources_or_conf)
    input_module = KnowledgeGraphEmbeddingInputModule(shared_resources)
    model_module = KnowledgeGraphEmbeddingModelModule(shared_resources, model_name='DistMult')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)


@link_prediction_reader
def complex_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """ Creates a knowledge_base_population Complex model."""
    from jack.readers.link_prediction.models import KnowledgeGraphEmbeddingInputModule, \
        KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule
    shared_resources = create_shared_resources(resources_or_conf)
    input_module = KnowledgeGraphEmbeddingInputModule(shared_resources)
    model_module = KnowledgeGraphEmbeddingModelModule(shared_resources, model_name='ComplEx')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)


@link_prediction_reader
def transe_reader(resources_or_conf: Union[dict, SharedResources] = None):
    """ Creates a knowledge_base_population TransE model."""
    from jack.readers.link_prediction.models import KnowledgeGraphEmbeddingInputModule, \
        KnowledgeGraphEmbeddingModelModule, \
        KnowledgeGraphEmbeddingOutputModule
    shared_resources = create_shared_resources(resources_or_conf)

    input_module = KnowledgeGraphEmbeddingInputModule(shared_resources)
    model_module = KnowledgeGraphEmbeddingModelModule(shared_resources, model_name='TransE')
    output_module = KnowledgeGraphEmbeddingOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)
