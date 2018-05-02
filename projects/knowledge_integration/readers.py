"""Reader definitions that use back"""

from jack.core.tensorflow import TFReader
from jack.readers.implementations import nli_reader, create_shared_resources, extractive_qa_reader


@extractive_qa_reader
def modular_assertion_qa_reader(resources_or_conf=None):
    from projects.knowledge_integration.qa.shared import XQAAssertionInputModule
    from jack.readers.extractive_qa.shared import XQAOutputModule
    from projects.knowledge_integration.qa.shared import ModularAssertionQAModel
    shared_resources = create_shared_resources(resources_or_conf)

    input_module = XQAAssertionInputModule(shared_resources)
    model_module = ModularAssertionQAModel(shared_resources)
    output_module = XQAOutputModule()
    return TFReader(shared_resources, input_module, model_module, output_module)


@extractive_qa_reader
def modular_assertion_definition_qa_reader(resources_or_conf=None):
    from projects.knowledge_integration.qa.definition_model import XQAAssertionDefinitionInputModule
    from projects.knowledge_integration.qa.definition_model import ModularAssertionDefinitionQAModel
    from jack.readers.extractive_qa.shared import XQAOutputModule
    shared_resources = create_shared_resources(resources_or_conf)

    input_module = XQAAssertionDefinitionInputModule(shared_resources)
    model_module = ModularAssertionDefinitionQAModel(shared_resources)
    output_module = XQAOutputModule()
    reader = TFReader(shared_resources, input_module, model_module, output_module)
    input_module.set_reader(reader)
    return TFReader(shared_resources, input_module, model_module, output_module)


@nli_reader
def cbilstm_nli_assertion_reader(resources_or_conf=None):
    from projects.knowledge_integration.nli import NLIAssertionModel
    from projects.knowledge_integration.nli import MultipleChoiceAssertionInputModule
    from jack.readers.classification.shared import SimpleClassificationOutputModule
    shared_resources = create_shared_resources(resources_or_conf)
    input_module = MultipleChoiceAssertionInputModule(shared_resources)
    model_module = NLIAssertionModel(shared_resources)
    output_module = SimpleClassificationOutputModule(shared_resources)
    return TFReader(shared_resources, input_module, model_module, output_module)
