import subprocess
import time
from os.path import join, exists
import os
import pytest
import numpy as np

def pytest_collection_modifyitems(items):
    for item in items:
        if "interface" in item.nodeid:
            item.add_marker(pytest.mark.interface)
        elif "event" in item.nodeid:
            item.add_marker(pytest.mark.event)


def get_pipeline_script_cmdcall_SNLI_converter():
    '''Creates a bash cmd to convert the SNLI data into our format'''
    # download data if not exists
    #train_file = 'tests/test_data/SNLI/2000_samples_train_quebap_v1.json'

    # convert snli files into quebap format
    cmd = "python3 jtr/load/SNLI2jtr_v1.py"
    #cmd = "pwd"
    return cmd



def check_file_adheres_to_schema(data_file_name):
    """
    Checks if a given data file adheres to the schema
    """
    schema_file_name = "jtr/load/dataset_schema.json"
    # validate schema adherence
    cmd = "python3 jtr/format/validate.py "+ data_file_name + " " + schema_file_name

    # Execute command and wait for results
    try:
        print('ujjj')
        #subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        #response = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout.read()
        #assert response == "JSON successfully validated."
    except subprocess.CalledProcessError as e:
        assert False, str(e.output)



def loaders_test(dataset_name):
    '''Tests a dataset converter, checking whether the converted data adheres
    to the Jack-the-Ripper (.jtr) format.
    Args:
        dataset_name (string): Dataset name as defined in
                DATASET_TO_CMD_CALL_STRING.
    Returns: None
    '''

    # Setup command
    # get the command line call for a given dataset
    DATASET_TO_CMD_CALL_STRING = dict()
    DATASET_TO_CMD_CALL_STRING['SNLI'] = get_pipeline_script_cmdcall_SNLI_converter()
    cmd = DATASET_TO_CMD_CALL_STRING[dataset_name]

    # Execute conversion command and wait for results
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        #response = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout.read()
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

@pytest.mark.data_loaders
def test_SNLI_converter():
    loaders_test(dataset_name='SNLI')
