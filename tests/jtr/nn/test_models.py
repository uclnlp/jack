# -*- coding: utf-8 -*-

import subprocess
import time
from os.path import join, exists
import os
import pytest
import numpy as np

OVERFIT_PATH = './tests/test_results/overfit_test/'
SMALLDATA_PATH = './tests/test_results/smalldata_test/'

models = [
        'snli_reader',
        'cbilstm_snli_reader',
        'dam_snli_reader',
        'esim_snli_reader'
    ]

# if you add a model here, you need the data in the format of:

# test_data/dataset-name/train.json
# test_data/dataset-name/dev.json
# test_data/dataset-name/test.json
# test_data/dataset-name/overfit.json

datasets = ['SNLI']
overfit_epochs = {'SNLI': 15}
small_data_epochs = {'SNLI': 5}

ids = []
testdata = []


def generate_test_data():
    '''Creates all permutations of models and datasets as tests.'''
    for dataset in datasets:
        for use_small_data in [False, True]:
            epochs = small_data_epochs[dataset] if use_small_data else overfit_epochs[dataset]
            for model in models:
                print(model)
                testdata.append([model, epochs, use_small_data, dataset])


def get_string_for_test(model_name, epochs, use_small_data, dataset):
    '''Creates a name for each test, so the output of PyTest is readable'''
    return ('model_name={0}, '
            'epochs={1}, run_type={2}, '
            'dataset={3}').format(model_name,
                                  epochs,
                                  ('smalldata' if use_small_data else 'overfit'), dataset)


def generate_names():
    '''Generates all names for all test cases'''
    for args in testdata:
        ids.append(get_string_for_test(*args))


generate_test_data()
generate_names()
print(testdata)


@pytest.mark.parametrize("model_name, epochs, use_small_data, dataset", testdata, ids=ids)
def test_model(model_name, epochs, use_small_data, dataset):
    '''Tests a model via training_pipeline.py by comparing with expected_result.txt
    Args:
        model_name (string): The model name as defined in the
                   training_pipeline.py dictionary.
        epochs (int=5): Some models need more time to overfit data; increase
               this the case of overfitting problems.
        use_small_data (bool=False): Switches between 'overfit' and 'smalldata'
                       mode.
        dataset (string): Dataset name as defined in
                DATASET_TO_CMD_CALL_STRING. This value is also used as the name
                to the test_result folder.
    Returns: None
    '''
    # Setup paths and filenames for the expected_results file
    test_result_path = join(SMALLDATA_PATH if use_small_data else OVERFIT_PATH, dataset, model_name)
    metric_filepath = join(test_result_path, datetime_test_result_filename())

    # create dir if it does not exists
    print(exists(test_result_path))
    if not exists(test_result_path):
        os.makedirs(test_result_path)

    # Stich together test data paths
    if not use_small_data:
        train_file = 'tests/test_data/{0}/overfit.json'.format(dataset)
        dev_file = train_file
        test_file = train_file
    else:
        train_file = 'tests/test_data/{0}/train.json'.format(dataset)
        dev_file = 'tests/test_data/{0}/dev.json'.format(dataset)
        test_file = 'tests/test_data/{0}/test.json'.format(dataset)

    # Setup the process call command
    cmd = 'CUDA_VISIBLE_DEVICES=-1 '  # we only test on the CPU
    cmd += "python3 jtr/train_reader.py with train={0} dev={1} test={2}".format(train_file, dev_file, test_file, )
    cmd += ' write_metrics_to={0}'.format(metric_filepath)
    cmd += ' model={0}'.format(model_name)
    cmd += ' epochs={0} learning_rate_decay=1.0'.format(epochs)
    print('command: ' + cmd)
    # Execute command and wait for results
    t0 = time.time()
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        for line in e.output.split(b'\n'):
            print(line)
        assert False, str(e.output)

    # Load and parse the results and the expected rults for testing
    new_results, runtime = load_and_parse_test_results(metric_filepath)
    expected_results, expected_runtime = load_and_parse_test_results(join(test_result_path,
                                                                          'expected_results.txt'))

    # Match expected results with current results; the order is important to
    # assert np.testing.assert_array_almost_equal(results[:,1], atol=0.01)

    for new, base in zip(new_results, expected_results):
        assert new[0] == base[0], "Different order of metrics!"
        assert np.allclose([new[1]], [base[1]], atol=0.05), \
            "Metric value different from expected results!"


def load_and_parse_test_results(filepath):
    '''This method loads and parses a metric file writen by EvalHook.'''
    name_value_metric_pair = []
    runtime = 0
    with open(filepath) as f:
        data = f.readlines()
        for i, line in enumerate(data):
            _date, _time, metric_name, metric_value = line.strip().split(' ')
            name_value_metric_pair.append([metric_name,
                                           np.float32(metric_value)])
    return name_value_metric_pair, runtime


def datetime_test_result_filename():
    '''Generates a string of the format testresult_CURRENT_DATE-TIME'''
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return 'testresult_' + timestr


test_model("snli_reader",1,False,"SNLI")
