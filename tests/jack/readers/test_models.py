# -*- coding: utf-8 -*-

import os
import subprocess
import time
from os.path import join, exists

import numpy as np
import pytest

OVERFIT_PATH = './tests/test_results/overfit_test/'
SMALLDATA_PATH = './tests/test_results/smalldata_test/'

# if you add a model here, you need the data in the format of:
# test_data/dataset-name/train.json
# test_data/dataset-name/dev.json
# test_data/dataset-name/test.json
# test_data/dataset-name/overfit.json

models2dataset = {}
models2dataset['esim'] = 'SNLI'
models2dataset['dam'] = 'SNLI'
models2dataset['fastqa'] = 'squad'
# models2dataset['jackqa'] = 'squad'

models2config = {}
models2config['esim'] = './conf/nli/esim.yaml'
models2config['dam'] = './conf/nli/dam.yaml'
models2config['fastqa'] = './conf/qa/fastqa.yaml'
models2config['jackqa'] = './conf/qa/jackqa.yaml'

overfit_epochs = {'SNLI': 15, 'squad': 15}
small_data_epochs = {'SNLI': 5, 'squad': 10}

modelspecifics = {
    'fastqa': lambda is_small_data: (' embedding_file=tests/test_data/glove.500.50d.txt' +
                                     ' embedding_format=glove' +
                                     ' vocab_from_embeddings=True'),
    'jackqa': lambda is_small_data: (' embedding_file=tests/test_data/glove.500.50d.txt' +
                                     ' embedding_format=glove' +
                                     ' vocab_from_embeddings=True'),
    'dam': lambda is_small_data: (' embedding_file=tests/test_data/glove.500.50d.txt' +
                                  ' embedding_format=glove' +
                                  ' vocab_from_embeddings=True'),
    'esim': lambda is_small_data: (' embedding_file=tests/test_data/glove.500.50d.txt' +
                                   ' embedding_format=glove' +
                                   ' vocab_from_embeddings=True'),
}

ids = []
testdata = []


def generate_test_data():
    '''Creates all permutations of models and datasets as tests.'''
    for reader, dataset in models2dataset.items():
        for use_small_data in [False, True]:
            epochs = small_data_epochs[dataset] if use_small_data else overfit_epochs[dataset]
            testdata.append([reader, epochs, use_small_data, dataset])


def get_string_for_test(reader, epochs, use_small_data, dataset):
    '''Creates a name for each test, so the output of PyTest is readable'''
    return ('reader={0}, '
            'epochs={1}, run_type={2}, '
            'dataset={3}').format(reader,
                                  epochs,
                                  ('smalldata' if use_small_data else 'overfit'), dataset)


def generate_names():
    '''Generates all names for all test cases'''
    for args in testdata:
        ids.append(get_string_for_test(*args))


generate_test_data()
generate_names()


@pytest.mark.parametrize("reader, epochs, use_small_data, dataset", testdata, ids=ids)
def test_model(reader, epochs, use_small_data, dataset):
    '''Tests a model via training_pipeline.py by comparing with expected_result.txt
    Args:
        reader (string): The reader
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
    test_result_path = join(SMALLDATA_PATH if use_small_data else OVERFIT_PATH, dataset, reader)
    metric_filepath = join(test_result_path, datetime_test_result_filename())

    # create dir if it does not exists
    if not exists(test_result_path):
        os.makedirs(test_result_path)

    # Stich together test data paths
    if not use_small_data:
        train_file = 'tests/test_data/{0}/overfit.json'.format(dataset)
        dev_file = train_file
        test_file = None
    else:
        train_file = 'tests/test_data/{0}/train.json'.format(dataset)
        dev_file = 'tests/test_data/{0}/dev.json'.format(dataset)
        test_file = None

    # Setup the process call command
    cmd = 'CUDA_VISIBLE_DEVICES=-1 '  # we only test on the CPU
    cmd += 'python3 ./bin/jack-train.py with'
    cmd += ' config={0}'.format(models2config[reader])
    cmd += ' train={0} dev={1} test={2}'.format(train_file, dev_file, test_file, )
    cmd += ' write_metrics_to={0}'.format(metric_filepath)
    cmd += ' epochs={0}'.format(epochs)
    cmd += ' learning_rate_decay=1.0'
    cmd += ' learning_rate=0.01'
    cmd += ' repr_dim=32'
    cmd += ' save_dir=None'
    cmd += ' validation_interval=-1'
    if reader in modelspecifics:
        # this is a function which takes use_small_data as argument
        cmd += modelspecifics[reader](use_small_data)
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
        assert np.allclose([new[1]], [base[1]], atol=0.1), \
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
