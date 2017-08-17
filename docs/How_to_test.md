# How to Test

## How Integration Tests Are Designed
Integration tests execute given models and given datasets using the [training_pipeline.py](/jtr/training_pipeline.py) with its default parameters for training. In addition to that however, an extra argument is passed to the training_pipeline method which switches on the [EvalHook](/jtr/util/hooks.py) write-metrics-to-disk behavior. Once a model run is completed the thus created metrics file (with metrics like accuracy or F1-score) is compared to a baseline.txt file which gives the target values for the metrics during a test run. This constitutes the integration test which is currently passed when both the order of the metrics and the metric values are the same (with a small absolute tolerance of 0.015).

A baseline file is created upon test creation and then serves as the needed model performance.

Currently, integration tests use truncated dataset files which are around 1MB in size (about 1-5k samples). These datasets are included in the [test_data](./tests/test_data) folder. The pipeline file itself is currently executed via an external process created from Python. Once this process is being executed the Python program waits until finishes and then proceeds with the checking the data.

## How to Invoke Tests
Tests can be invoked by using the Makefile in the main jtr directory. There are several commands which execute different test batteries.

|  Description                      | make command                                                 |
|-----------------------------------|--------------------------------------------------------------|
| Run all unit tests                | `make test`                                                  |
| Run all overfit integration tests | `make test-overfit`                                               |
| Run all integration tests than are ran with small data samples (<2k samples)| `make test-smalldata`   |
| Run all sentihood integration test| `make test-sentihood`                                             |
| Run all SNLI integration test| `make test-SNLI`                                                       |
| Run all all GPU tests | `make test-GPU`                                                       |


To run a test battery with the GPU simply add "gpu" to the make command, for example`make overfitgpu` executes all overfit integration tests on the GPU. Please note if you do not have a GPU the code is executed on the CPU and compares the results with GPU baselines which will most likely yield an error.

For more differentiated test execution you should run pytest directly from the main (first) jtr directory. The core command for this is `pytest -v -m "(flag1 and flag2 and not flag3)"`. You find some examples below.

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

You can find existing integration test in [test_models.py](/jtr/test_models.py).

#### 1. Create test data (if you add a new dataset)
- Currently test data is about 1 MB for each file which can hold about 1-5k samples.
- An "overfit" file contains just 100 samples as a quick test to overfit this data.
- **Important**: Dataset naming conventions; for testing store your data in tests/test_data/DATASET_NAME/ under the following names
  - *overfit.json* for the overfit tests
  - *DATASET_NAME-train.json* for the training set (about 1k samples) 
  - *DATASET_NAME-dev.json* for the development set (about 1k samples) 
  - *DATASET_NAME-test.json* for the test set (about 1k samples) 

#### 2. Add test cases
The tests cases are automatically created once you add a model or a dataset, which means if you add a model, it will be tested on all dataset and on both the CPU and GPU. If you add a dataset, then tests for all models and both the CPU and GPU will be created.
- You can add a new dataset by simply adding a line into the dataset_epochs list; the second entry of the tuple denotes the number of epochs to run the tests, for example to add your dataset with name 'CoolDataset' and run tests for 5 epochs add `('CoolDataset', 5)` to the list
- You can add a new model by adding it to the models list. In addition to that you will need to add the same model name in the dictionary in the [training_pipeline.py](/jtr/training_pipeline.py) file (search for models)
  
#### 5. Create baselines
- Run your model for both the CPU and GPU, check the generated files in the [test_results](./test_results) folder. If the results seem plausible rename the files to expected_results.txt
- Rerun your model, it should now pass the tests
- Commit your work, you are done!

#### 6. Edit `tests/conftest.py`
- Explanation: This file is responsible to make it possible to filter test according their name, for example the "test-sentihood" in`pytest -v -m sentihood`
- If you added a dataset and all test mosts contain the dataset name (they should do) then add it into the filter (take the other data sets as an example)
- If you added a model you can also create a filter for that (currently no such filter exists as an example)
