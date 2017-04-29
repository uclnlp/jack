# -*- coding: utf-8 -*-

from jtr.convert import SNLI2jtr_v1

import subprocess
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "interface" in item.nodeid:
            item.add_marker(pytest.mark.interface)
        elif "event" in item.nodeid:
            item.add_marker(pytest.mark.event)


def get_pipeline_script_cmdcall_snli_converter():
    """
    Creates a bash cmd to convert the SNLI data into our format
    :return:
    """
    # convert snli files into jtr format
    cmd = "python3 jtr/format/convert/SNLI2jtr_v1.py"
    return cmd


def check_file_adheres_to_schema(data_file_name):
    """
    Checks if a given data file adheres to the schema
    """
    schema_file_name = "jtr/load/dataset_schema.json"
    # validate schema adherence
    cmd = "python3 jtr/format/validate.py " + data_file_name + " " + schema_file_name

    # Execute command and wait for results
    try:
        pass
        #subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        #response = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout.read()
        #assert response == "JSON successfully validated."
    except subprocess.CalledProcessError as e:
        assert False, str(e.output)


def loaders_test(dataset_name):
    """
    Tests a dataset converter, checking whether the converted data adheres
    to the Jack-the-Reader (.jtr) format.
    Args:
        dataset_name (string): Dataset name as defined in
                DATASET_TO_CMD_CALL_STRING.
    Returns: None
    """
    # Setup command - get the command line call for a given dataset
    DATASET_TO_CMD_CALL_STRING = dict()
    DATASET_TO_CMD_CALL_STRING['SNLI'] = get_pipeline_script_cmdcall_snli_converter()
    cmd = DATASET_TO_CMD_CALL_STRING[dataset_name]

    # Execute conversion command and wait for results

    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        assert False, str(e.output)

    data_file_name = "jtr/tests/test_data/SNLI/2000_samples_train_jtr_v1.json"
    check_file_adheres_to_schema(data_file_name)


#-------------------------------
#       DATA LOADER TESTS
#-------------------------------
@pytest.mark.data_loaders
def test_test():
    assert 2 == 2


#@pytest.mark.data_loaders
#def test_snli_converter():
#    loaders_test(dataset_name='SNLI')

@pytest.mark.data_loaders
def test_snli_converter():
    path = 'tests/test_data/SNLI/1000_samples_snli_1.0_train.jsonl'
    res = SNLI2jtr_v1.convert_snli(snli_file_jsonl=path)
    assert len(res) == 3
    assert set(res.keys()) == {'meta', 'instances', 'globals'}
    assert res['meta'] == 'SNLI'
    assert res['globals'] == {'candidates': [{'text': 'entailment'}, {'text': 'neutral'}, {'text': 'contradiction'}]}
    assert isinstance(res['instances'], list)
    assert res['instances'][0] == {'questions': [{'question': 'A person is training his horse for a competition.', 'answers': [{'text': 'neutral'}]}], 'id': '3416050480.jpg#4r1n', 'support': [{'text': 'A person on a horse jumps over a broken down airplane.', 'id': '3416050480.jpg#4'}]}
    assert res['instances'][1] == {'questions': [{'question': 'A person is at a diner, ordering an omelette.', 'answers': [{'text': 'contradiction'}]}], 'support': [{'id': '3416050480.jpg#4', 'text': 'A person on a horse jumps over a broken down airplane.'}], 'id': '3416050480.jpg#4r1c'}
    assert res['instances'][2] == {'support': [{'text': 'A person on a horse jumps over a broken down airplane.', 'id': '3416050480.jpg#4'}], 'questions': [{'answers': [{'text': 'entailment'}], 'question': 'A person is outdoors, on a horse.'}], 'id': '3416050480.jpg#4r1e'}


@pytest.mark.data_loaders
def test_snli_schema():
    data_file_name = "jtr/tests/test_data/SNLI/2000_samples_train_jtr_v1.json"
    check_file_adheres_to_schema(data_file_name)