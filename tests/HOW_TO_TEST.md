# How to Test

## How Integration Tests Are Designed
Integration tests execute given models and given datasets using the [training_pipeline.py](../quebap/training_pipeline.py) with its default parameters for training. In addition to that however, an extra argument is passed to the training_pipeline method which switches on the [EvalHook](../quebap/sisyphos/hooks.py) write-metrics-to-disk behavior. Once a model run is completed the thus created metrics file (with metrics like accuracy or F1-score) is compared to a baseline.txt file which gives the target values for the metrics during a test run. This constitutes the integration test which is currently passed when both the order of the metrics and the metric values are the same (with a small absolute tolerance of 0.015).

A baseline file is created upon test creation and then serves as the needed model performance.

Currently, integration tests use truncated dataset files which are around 1MB in size (about 1-5k samples). These datasets are included in the [test_data](./test_data) folder. The pipeline file itself is currently executed via an external process created from Python. Once this process is being executed the Python program waits until finishes and then proceeds with the checking the data.

## How to Invoke Tests
Tests can be invoked by using the Makefile in the main quebap directory. There are several commands which execute different test batteries.

|  Description                      | make command                                                 |
|-----------------------------------|--------------------------------------------------------------|
| Run all unit tests                | `make test`                                                  |
| Run all overfit integration tests | `make overfit`                                               |
| Run all integration tests than are ran with small data samples (<2k samples)| `make smalldata`   |
| Run all sentihood integration test| `make sentihood`                                             |
| Run all SNLI integration test| `make SNLI`                                                       |


To run a test battery with the GPU simply add "gpu" to the make command, for example`make overfitgpu` executes all overfit integration tests on the GPU. Please note if you do not have a GPU the code is executed on the CPU and compares the results with GPU baselines which will most likely yield an error.

For more differentiated test execution you should run pytest directly from the main (first) quebap directory. The core command for this is `pytest -v -m "(flag1 and flag2 and not flag3)"`. You find some examples below.

|  Description                                 | make command                                                 |
|----------------------------------------------|--------------------------------------------------------------|
| Run all sentihood dataset test               | `pytest -v -m sentihood`                                     |
| Run all sentihood dataset GPU tests          | `pytest -v -m "(sentihood and GPU)"`                         |
| Run all sentihood dataset GPU overfit tests  | `pytest -v -m "(sentihood and GPU and overfit)"`             | 

Unit tests are usually very fast, however the integration tests need quite some time even though they operate on small, truncated dataset. Current runtimes on my laptop (CPU) and desktop (GPU: GTX Titan X):
  - SNLI CPU:      Overfit: 37 sec; Small data: 462 sec
  - SNLI GPU:      Overfit: 55 sec; Small data: 563 sec
  - sentihood GPU: Overfit: 36; Small data: 144 sec 
  - sentihood CPU: Overfit: 56; Small data: 100 sec 

## How to Add Integration Tests

You can find existing integration test in [test_models.py](/quebap/test_models.py).

#### 1. Create test data
- Currently test data is about 1 MB for each file which can hold about 1-5k samples.
- An "overfit" file contains just 100 samples as a quick test to overfit this data.

#### 2. Add data loading methods
- Have a look at `get_pipeline_script_cmdcall_SNLI_overfit()` and `get_pipeline_script_cmdcall_SNLI_smalldata()` and follow their design to create a command line call in string format

#### 3. Add your loading methods to the dataset dictionary
- Search for `DATASET_TO_CMD_CALL_STRING` in the file and add your methods in the same pattern. The design of this piece of code may seem questionable at first, but with adding the dataset loading methods to this dictionary we avoid making any changes to the main integration test method `test_model(..)`

#### 4. Create a new test stub and call `test_model(..)`
- Create a test function at the appropriate space (their are regions for GPU/CPU and overfit vs smalldata sets)
- decorate your function with the right pytest mark; you want two tests, one for the CPU and one for the GPU
- Call `test_model(..)`:
  - For GPU tests, call `test_model(.., useGPUID=0, ..)`. The `useGPUID=0` parameter means that the first available GPU is used for this test.
  - For CPU tests, call `test_model(.., useGPUID=-1, ..)`. The `useGPUID=-1` parameter means that no GPU is used for this test.
  - Use `use_small_data=False` for overfit tests, `use_small_data=True` for smalldata tests
  - Use `dataset='YOUR DATASET NAME'`
  
#### 5. Create baselines
- Run your model for both the CPU and GPU, check the generated files in the [test_results](./test_results) folder. If the results seem plausible rename the files to baseline.txt
- Rerun your model, it should now pass the tests
- Commit your work, you are done!

#### 6. Edit [conftest.py](conftest.py)
- Explanation: This file is responsible to make it possible to filter test according their name, for example the "sentihood" in`pytest -v -m sentihood`
- If you added a dataset and all test mosts contain the dataset name (they should do) then add it into the filter (take the other data sets as an example)
- If you added a model you can also create a filter for that (currently no such filter exists as an example)
