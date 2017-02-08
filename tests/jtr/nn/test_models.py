import subprocess
import time
from os.path import join, exists
import os
import pytest
import numpy as np

OVERFIT_PATH_CPU = './tests/test_results/overfit_test/CPU/'
OVERFIT_PATH_GPU = './tests/test_results/overfit_test/GPU/'
SMALLDATA_PATH_CPU = './tests/test_results/smalldata_test/CPU/'
SMALLDATA_PATH_GPU = './tests/test_results/smalldata_test/GPU/'

# the runtime factor gives the approximate speed of the GPU relative
# to the baseline of a GTX Titan X
GPU_MODELS_RUNTIME_FACTOR = {}
GPU_MODELS_RUNTIME_FACTOR['GeForce GTX Titan X'] = 1.0

models = \
[
    'boe_nosupport',
    'boe_support',
    'bicond_singlesupport_reader',
    'bicond_singlesupport_reader_with_cands',
    'bilstm_singlesupport_reader_with_cands',
    'bilstm_nosupport_reader_with_cands'
]

# if you add a model here, you need the data in the format of:

# test_data/dataset-name/dataset_name-train.json
# test_data/dataset-name/dataset_name-dev.json
# test_data/dataset-name/dataset_name-test.json
# test_data/dataset-name/overfit.json

datasets = ['SNLI', 'sentihood']
overfit_epochs = {'SNLI': 15, 'sentihood': 15}
small_data_epochs = {'SNLI': 5, 'sentihood': 3}

ids = []
testdata = []

def generate_test_data():
    '''Creates all permutations of models and datasets as tests.'''
    for dataset in datasets:
        for useGPUID in [-1, 0]:
            for use_small_data in [False, True]:
                epochs = small_data_epochs[dataset] if use_small_data else overfit_epochs[dataset]
                for model in models:
                    testdata.append([model, useGPUID, epochs, use_small_data, \
                        dataset])


def get_string_for_test(model_name, useGPUID, epochs, use_small_data, dataset):
    '''Creates a name for each test, so the output of PyTest is readable'''
    return ('model_name={0}, {1}, '
             'epochs={2}, run_type={3}, '
             'dataset={4}').format(model_name,
                ('GPU' if useGPUID >= 0 else 'CPU'), epochs,
                ('smalldata' if use_small_data else 'overfit'), dataset)

def generate_names():
    '''Generates all names for all test cases'''
    for args in testdata:
        ids.append(get_string_for_test(*args))

generate_test_data()
generate_names()

@pytest.mark.parametrize("model_name, useGPUID, epochs, use_small_data, \
        dataset", testdata, ids=ids)
def test_model(model_name, useGPUID, epochs, use_small_data,
        dataset):
    '''Tests a model via training_pipeline.py by comparing with expected_result.txt
    Args:
        model_name (string): The model name as defined in the
                   training_pipeline.py dictionary.
        useGPUID (int): -1 means the CPU is used; > -1 means the GPU is used,
                 that is if a GPU is available. NOTE: if > -1 always compares
                 against GPU expected results, even if the CPU is used.
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
    if use_small_data and useGPUID >= 0:
        test_result_path = join(SMALLDATA_PATH_GPU, dataset, model_name)
    elif use_small_data and useGPUID == -1:
        test_result_path = join(SMALLDATA_PATH_CPU, dataset, model_name)
    elif not use_small_data and useGPUID >= 0:
        test_result_path = join(OVERFIT_PATH_GPU, dataset, model_name)
    elif not use_small_data and useGPUID == -1:
        test_result_path = join(OVERFIT_PATH_CPU, dataset, model_name)

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
        train_file = 'tests/test_data/{0}/{0}-train.json'.format(dataset)
        dev_file  = 'tests/test_data/{0}/{0}-dev.json'.format(dataset)
        test_file  = 'tests/test_data/{0}/{0}-test.json'.format(dataset)

    # Setup the process call command
    cmd = 'CUDA_VISIBLE_DEVICES={0} '.format(useGPUID)
    cmd += "python3 jtr/training_pipeline.py --train={0} --dev={1} \
    --test={2}" .format(train_file, dev_file, test_file,)
    cmd += ' --write_metrics_to={0}'.format(metric_filepath)
    cmd += ' --model={0}'.format(model_name)
    cmd += ' --epochs={0}'.format(epochs)
    print('command: '+cmd)
    # Execute command and wait for results
    t0 = time.time()
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        for line in e.output.split(b'\n'):
            print(line)
        assert False, str(e.output)
    runtime = time.time()-t0

    # add runtime information for GPU runs
    if useGPUID>=0:
        with open(metric_filepath, 'a') as f:
            f.write('{0}\n'.format(np.round(runtime)))

    # Load and parse the results and the expected rults for testing
    new_results, runtime = load_and_parse_test_results(metric_filepath,
            useGPUID>=0)
    expected_results, expected_runtime = load_and_parse_test_results(join(test_result_path,
        'expected_results.txt'), useGPUID>=0)

    # Match expected results with current results; the order is important to
    #assert np.testing.assert_array_almost_equal(results[:,1], atol=0.01)

    for new, base in zip(new_results, expected_results):
        assert new[0] == base[0], "Different order of metrics!"
        assert np.allclose([new[1]],[base[1]],atol=0.015), \
            "Metric value different from expected results!"

    if useGPUID>=0:
        # We check only GPU runtime and weight it by a GPU runtime weight
        # where faster GPUs need less time. CPU runtime is too difficult to compare
        # and is neglected at this point.
        GPU_names = subprocess.check_output( \
            "nvidia-smi --query-gpu=gpu_name --format=csv,noheader", shell=True)
        # we use the GPU at ID=0 for testing, that is index 0 of what is returned
        # by nvidia-smi
        GPU0_name = GPU_names.strip().split(b'\n')[0]
        if GPU0_name in GPU_MODELS_RUNTIME_FACTOR:
            factor = GPU_MODELS_RUNTIME_FACTOR[GPU0_name]
            assert np.allclose([runtime],[expected_runtime/factor],rtol=0.05), \
                "Runtime performance is off by more than 5%"


def load_and_parse_test_results(filepath, useGPU=False):
    '''This method loads and parses a metric file writen by EvalHook.'''
    name_value_metric_pair = []
    runtime = 0
    with open(filepath) as f:
        data = f.readlines()
        n = len(data)
        for i, line in enumerate(data):
            if i == n-1 and useGPU:
                runtime = np.float32(line.strip())
            else:
                _date, _time, metric_name, metric_value = line.strip().split(' ')
                name_value_metric_pair.append([metric_name,
                    np.float32(metric_value)])
    return name_value_metric_pair, runtime

def datetime_test_result_filename():
    '''Generates a string of the format testresult_CURRENT_DATE-TIME'''
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return 'testresult_' + timestr

