# -*- coding: utf-8 -*-

import subprocess
import time
from os.path import join, exists
import os
import pytest
import numpy as np
import copy
import tensorflow as tf

from jack.train_reader import train
from jack.io.load import loaders
from jack.io.embeddings.embeddings import load_embeddings, Embeddings
from jack.util.vocab import Vocab
from jack.core.shared_resources import SharedResources
from jack import readers

OVERFIT_PATH = './tests/test_results/overfit_test/'
SMALLDATA_PATH = './tests/test_results/smalldata_test/'

# if you add a model here, you need the data in the format of:
# test_data/dataset-name/train.json
# test_data/dataset-name/dev.json
# test_data/dataset-name/test.json
# test_data/dataset-name/overfit.json

general_config = {
        'seed': 1337,
        'clip_value': 0.0,
        'batch_size': 128,
        'epochs': 10,
        'l2': 0.0,
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'learning_rate_decay': 0.5,
        'log_interval': 100,
        'validation_interval': None,
        'tensorboard_folder': None,
        'model': None,
        'model_dir': None,
        'write_metrics_to': None
    }

# these are general settings for the model function
models2dataset = {}
models2dataset['dam_snli_reader'] = 'SNLI'
models2dataset['esim_snli_reader'] = 'SNLI'
models2dataset['cbilstm_snli_reader'] = 'SNLI'
models2dataset['fastqa_reader'] = 'squad'
models2dataset['bidaf_reader'] = 'squad'

overfit_epochs = {'SNLI': 15, 'SNLI_stream' : 15, 'squad': 15}
small_data_epochs = {'SNLI': 5, 'SNLI_stream' : 5, 'squad': 10}

# The model_config is a mix of shared config and parameters that are usually passed via
# command line. Add shared parameters here. If you need to add a different command line
# parameter to make your model work, you need to adjust the code in the test_model function.
model_config = {}
default_params = {}
default_params['repr_dim'] = 128
default_params['repr_dim_input'] = 128
default_params['dropout'] = 0.0

model_config['dam_snli_reader'] = default_params
model_config['esim_snli_reader'] = default_params
model_config['cbilstm_snli_reader'] = default_params

model_config['fastqa_reader'] = {}
model_config['fastqa_reader']['repr_dim'] = 32
model_config['fastqa_reader']['repr_dim_input'] = 50
model_config['fastqa_reader']['pretrain'] = True
model_config['fastqa_reader']['embedding_file'] = 'tests/test_data/glove.500.50d.txt'
model_config['fastqa_reader']['embedding_format'] = 'glove'

model_config['bidaf_reader'] = copy.deepcopy(model_config['fastqa_reader'])
model_config['bidaf_reader']['epochs'] = (2, 5) # for (small_data, overfit) epochs

ids = []
testdata = []

def generate_test_data():
    '''Creates all permutations of models and datasets as tests.'''
    for model, dataset in models2dataset.items():
        for use_small_data in [False, True]:
            epochs = small_data_epochs[dataset] if use_small_data else overfit_epochs[dataset]
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
    with tf.variable_scope(model_name + '_' + str(use_small_data)):
        # Setup paths and filenames for the expected_results file
        test_result_path = join(SMALLDATA_PATH if use_small_data else OVERFIT_PATH, dataset, model_name)
        metric_filepath = join(test_result_path, datetime_test_result_filename())

        # create dir if it does not exists
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


        # copy global config and change required parameters
        local_config = copy.deepcopy(general_config)
        local_config['write_metrics_to'] = metric_filepath
        local_config['epochs'] = epochs
        local_config['model'] = model_name

        # write specific epochs for small data/overfit
        shared_config = model_config[model_name]
        if 'epochs' in shared_config:
            epochs = shared_config.pop('epochs')
            local_config['epochs'] = epochs[0] if use_small_data else epochs[1]


        # load data
        train_data = loaders['jack'](train_file)
        dev_data = loaders['jack'](dev_file)
        test_data = loaders['jack'](test_file)

        # load embeddings
        if 'pretrain' in shared_config:
            embeddings = load_embeddings(shared_config['embedding_file'], shared_config['embedding_format'])
            shared_config["repr_dim_input"] = embeddings.lookup[0].shape[0]
        else:
            embeddings = Embeddings(None, None)

        emb = embeddings

        vocab = Vocab(emb=emb, init_from_embeddings=False)

        shared_resources = SharedResources(vocab, shared_config)
        reader = readers.readers[model_name](shared_resources)

        # train
        train(reader, train_data, test_data, dev_data, local_config)


        # Load and parse the results and the expected rults for testing
        new_results, runtime = load_and_parse_test_results(metric_filepath)
        expected_results, expected_runtime = load_and_parse_test_results(join(test_result_path,
                                                                          'expected_results.txt'))
    # nuke the graph, so that the next test on the same thread does not
    # run into a "variable already exists" error
    tf.reset_default_graph()

    # Match expected results with current results; the order is important to
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
