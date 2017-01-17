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

def load_and_parse_test_results(filepath):
    '''This method loads and parses a metric file writen by EvalHook.'''
    name_value_metric_pair = []
    with open(filepath) as f:
        for line in f:
            print(line)
            _date, _time, metric_name, metric_value = line.strip().split(' ')
            name_value_metric_pair.append([metric_name,
                np.float32(metric_value)])
    return name_value_metric_pair

def datetime_test_result_filename():
    '''Generates a string of the format testresult_CURRENT_DATE-TIME'''
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return 'testresult_' + timestr

def get_pipeline_script_cmdcall_SNLI_overfit():
    '''Creates a bash cmd with overfit data to invoke the pipeline script'''
    train_file = 'quebap/data/SNLI/snippet_quebapformat_v1.json'
    dev_file = train_file
    test_file = train_file
    # Setup command
    cmd = "python3 quebap/training_pipeline.py --train={0} --dev={1} \
    --test={2}" .format(train_file, dev_file, test_file,)
    return cmd

def get_pipeline_script_cmdcall_sentihood_overfit():
    train_file = 'tests/test_data/sentihood/overfit.json'
    dev_file = train_file
    test_file = train_file
    # Setup command
    cmd = "python3 quebap/training_pipeline.py --train={0} --dev={1} \
    --test={2}" .format(train_file, dev_file, test_file,)
    return cmd

def get_pipeline_script_cmdcall_SNLI_smalldata():
    '''Creates a bash cmd with a small dataset to invoke the pipeline script'''
    train_file = 'tests/test_data/SNLI/2000_samples_train_quebap_v1.json'
    dev_file  = 'tests/test_data/SNLI/1000_samples_dev_quebap_v1.json'
    test_file  = 'tests/test_data/SNLI/2000_samples_test_quebap_v1.json'
    # Setup command
    cmd = "python3 quebap/training_pipeline.py --train={0} --dev={1} \
    --test={2}" .format(train_file, dev_file, test_file,)
    return cmd

def get_pipeline_script_cmdcall_sentihood_smalldata():
    train_file = 'tests/test_data/sentihood/sentihood_train.json'
    dev_file  = 'tests/test_data/sentihood/sentihood_dev.json'
    test_file  = 'tests/test_data/sentihood/sentihood_test.json'
    # Setup command
    cmd = "python3 quebap/training_pipeline.py --train={0} --dev={1} \
    --test={2}" .format(train_file, dev_file, test_file,)
    return cmd

# This dictionary is here so that the model_test method does not need
# to be changed when new datasets are added. The only change needed
# will be to add an entry to this dictionary.

DATASET_TO_CMD_CALL_STRING = {}
DATASET_TO_CMD_CALL_STRING ['sentihood'] = \
(
        (get_pipeline_script_cmdcall_sentihood_overfit(),
        get_pipeline_script_cmdcall_sentihood_smalldata())
)
DATASET_TO_CMD_CALL_STRING ['SNLI'] = \
(
        (get_pipeline_script_cmdcall_SNLI_overfit(),
        get_pipeline_script_cmdcall_SNLI_smalldata())
)

def model_test(model_name, useGPUID=-1, epochs=5, use_small_data=False,
        dataset='SNLI'):
    '''Tests a model via training_pipeline.py by comparing with baseline.txt
    Args:
        model_name (string): The model name as defined in the
                   training_pipeline.py dictionary.
        useGPUID (int): -1 means the CPU is used; > -1 means the GPU is used,
                 that is if a GPU is available. NOTE: if > -1 always compares
                 against GPU baselines, even if the CPU is used.
        epochs (int=5): Some models need more time to overfit data; increase
               this the case of overfitting problems.
        use_small_data (bool=False): Switches between 'overfit' and 'smalldata'
                       mode.
        dataset (string): Dataset name as defined in
                DATASET_TO_CMD_CALL_STRING. This value is also used as the name
                to the test_result folder.
    Returns: None
    '''
    # Setup paths and filenames
    if use_small_data and useGPUID >= 0:
        test_result_path = join(SMALLDATA_PATH_GPU, model_name + '_' + dataset)
    elif use_small_data and useGPUID == -1:
        test_result_path = join(SMALLDATA_PATH_CPU, model_name + '_' + dataset)
    elif not use_small_data and useGPUID >= 0:
        test_result_path = join(OVERFIT_PATH_GPU, model_name + '_' + dataset)
    elif not use_small_data and useGPUID == -1:
        test_result_path = join(OVERFIT_PATH_CPU, model_name + '_' + dataset)

    metric_filepath = join(test_result_path, datetime_test_result_filename())

    # create dir if it does not exists
    if not exists(test_result_path): os.mkdir(test_result_path)

    # Setup command
    cmd = ""
    if use_small_data:
        cmd = DATASET_TO_CMD_CALL_STRING[dataset][1]
    else:
        cmd = DATASET_TO_CMD_CALL_STRING[dataset][0]
    cmd += ' --write_metrics_to={0}'.format(metric_filepath)
    cmd += ' --model={0}'.format(model_name)
    cmd += ' --epochs={0}'.format(epochs)

    cmd = 'CUDA_VISIBLE_DEVICES={0} '.format(useGPUID) + cmd

    # Execute command and wait for results
    p = subprocess.Popen(cmd, shell=True)
    p.communicate() # blocking

    # Load and parse the results and the baseline for testing
    new_results = load_and_parse_test_results(metric_filepath)
    baseline = load_and_parse_test_results(join(test_result_path,
        'baseline.txt'))

    # Match baseline with current results; the order is important to
    #assert np.testing.assert_array_almost_equal(results[:,1], atol=0.01)

    for new, base in zip(new_results, baseline):
        assert new[0] == base[0], "Different order of metrics!"
        assert np.allclose([new[1]],[base[1]],atol=0.015), "Metric value different from baseline!"

#-------------------------------
#       CPU OVERFIT TESTS
#-------------------------------


#-------------------------------
#           SNLI
#-------------------------------

@pytest.mark.overfit
def test_biconditional_reader_SNLI_overfit():
    model_test('bicond_singlesupport_reader')

@pytest.mark.overfit
def test_biconditional_reader_with_candidates_SNLI_overfit():
    model_test('bicond_singlesupport_reader_with_cands')

@pytest.mark.overfit
def test_bilstm_reader_with_candidates_SNLI_overfit():
    model_test('bilstm_singlesupport_reader_with_cands')

@pytest.mark.overfit
def test_bilstm_reader_with_candidates_no_support_SNLI_overfit():
    model_test('bilstm_nosupport_reader_with_cands')

@pytest.mark.overfit
def test_bag_of_embeddings_with_support_SNLI_overfit():
    model_test('boe_support')

@pytest.mark.overfit
def test_bag_of_embeddings_no_support_SNLI_overfit():
    model_test('boe_nosupport')

#-------------------------------
#           sentihood
#-------------------------------

@pytest.mark.overfit
def test_biconditional_reader_sentihood_overfit():
    model_test('boe_nosupport', dataset='sentihood')

#-------------------------------
#       GPU OVERFIT TESTS
#-------------------------------

@pytest.mark.overfitgpu
def test_biconditional_reader_SNLI_overfit_GPU():
    model_test('bicond_singlesupport_reader', useGPUID=0)

@pytest.mark.overfitgpu
def test_biconditional_reader_with_candidates_SNLI_overfit_GPU():
    model_test('bicond_singlesupport_reader_with_cands', useGPUID=0)

@pytest.mark.overfitgpu
def test_bilstm_reader_with_candidates_SNLI_overfit_GPU():
    model_test('bilstm_singlesupport_reader_with_cands', useGPUID=0)

@pytest.mark.overfitgpu
def test_bilstm_reader_with_candidates_no_support_SNLI_overfit_GPU():
    model_test('bilstm_nosupport_reader_with_cands', useGPUID=0)

@pytest.mark.overfitgpu
def test_bag_of_embeddings_with_support_SNLI_overfit_GPU():
    model_test('boe_support', useGPUID=0)

@pytest.mark.overfitgpu
def test_bag_of_embeddings_no_support_SNLI_overfit_GPU():
    model_test('boe_nosupport', useGPUID=0)

#-------------------------------
#       CPU SMALLDATA TESTS
#-------------------------------

@pytest.mark.smalldata
def test_biconditional_reader_SNLI_smalldata():
    model_test('bicond_singlesupport_reader', use_small_data=True)

@pytest.mark.smalldata
def test_biconditional_reader_with_candidates_SNLI_smalldata():
    model_test('bicond_singlesupport_reader_with_cands',
            use_small_data=True)

@pytest.mark.smalldata
def test_bilstm_reader_with_candidates_SNLI_smalldata():
    model_test('bilstm_singlesupport_reader_with_cands',
            use_small_data=True)

@pytest.mark.smalldata
def test_bilstm_reader_with_candidates_no_support_SNLI_smalldata():
    model_test('bilstm_nosupport_reader_with_cands', use_small_data=True)

@pytest.mark.smalldata
def test_bag_of_embeddings_with_support_SNLI_smalldata():
    model_test('boe_support', use_small_data=True)

#-------------------------------
#       GPU SMALLDATA TESTS
#-------------------------------


@pytest.mark.smalldatagpu
def test_biconditional_reader_SNLI_smalldata_GPU():
    model_test('bicond_singlesupport_reader',
            use_small_data=True,useGPUID=0)

@pytest.mark.smalldatagpu
def test_biconditional_reader_with_candidates_SNLI_smalldata_GPU():
    model_test('bicond_singlesupport_reader_with_cands',
            use_small_data=True,useGPUID=0)

@pytest.mark.smalldatagpu
def test_bilstm_reader_with_candidates_SNLI_smalldata_GPU():
    model_test('bilstm_singlesupport_reader_with_cands',
            use_small_data=True, useGPUID=0)

@pytest.mark.smalldatagpu
def test_bilstm_reader_with_candidates_no_support_SNLI_smalldata_GPU():
    model_test('bilstm_nosupport_reader_with_cands',
            use_small_data=True, useGPUID=0)

@pytest.mark.smalldatagpu
def test_bag_of_embeddings_with_support_SNLI_smalldata_GPU():
    model_test('boe_support', use_small_data=True, useGPUID=0)
