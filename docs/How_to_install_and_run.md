# How to Install and Run Quebap

## Install Quebap and test your installation

The installing procedure currently has three plus one steps:
  1. Install Tensorflow
  2. Run [setup.py](../setup.py)
  3. Test your install
  4. (Optional) install missing dependencies

#### 1. Install Tensorflow
  - Requirements: Python3, [pip3](http://stackoverflow.com/questions/6587507/how-to-install-pip-with-python-3)
  - GPU requirements: [CUDA 8.0](https://developer.nvidia.com/cuda-downloads) and cuDNN (you need to register with NVIDIA)
  - Follow the TensorFlow [installation guide](https://www.tensorflow.org/get_started/os_setup)

#### 2. Run [setup.py](../setup.py)
  - Run in the main directory: `sudo python setup.py install`

#### 3. If you run in some problems with missing libraries
  - Run in the main directory `make test` to test the core functionality of Quebap
  - Run in the main directory `make overfit` to test the functionality of the integration between TensorFlow and Quebap

#### 4. (Optional) Install missing requirements
  - In some cases you might to manually install some requirements to fix some errors during the `make overfit` procedure. You can install these requirements with `pip3 install -r requirement.txt` in the main directory

## Run an example in Quebap

There are currently four steps to get an example running with Quebap:
  1. Download data via script
  2. Convert data with preprocessing script into the Quebap JSON format
  3. Run training_pipeline.py with parameters for your dataset and the model that you want
  4. You can find the list of available models in the models dictionary in the beginning of the training_pipeline.py file

#### 1. Download data via script
  - Have a look to quebap/data to get an overview over the currently available dataset
  - Download the data by running the download.sh script (not all dataset currently have such a script), for example for SNLI cd into its directory and run `sh download.sh`

#### 2. Convert data with preprocessing script into the Quebap JSON format
  - Go to the main quebap directory and execute the preprocessing script located in quebap/load for example for SNLI you can run `python3 quebap/load/SNLI2quebap_v1.py` the data will be converted and rewritten into the quebap/data directory

#### 3. Run training_pipeline.py with parameters for your dataset and the model that you want
  - The main point of entry to run a model with Quebap is [training_pipeline.py](../quebap/training_pipeline.py) we need to pass some parameters for our data into the script along with the model that we want to run. By default we run a pair of conditional bidirectional LSTM readers that make use of a single supporting information (input: Question; support: Supporting document which should be used to answer that question)
  - For the data the pipeline expects 3 filepath arguments to Quebap JSON files: train, dev, and test for the respective data sets. For example for SNLI we have: 
  ```
  python3 training_pipeline.py --train=data/SNLI/snli_1.0/snli_1.0_train_quebap_v1.json --dev=data/SNLI/snli_1.0/snli_1.0_dev_quebap_v1.json --test=data/SNLI/snli_1.0/snli_1.0_test_quebap_v1.json
  ```
  - At this point we can also add other parameters like learning rate and L2 penalty: 
  ```
  python3 training_pipeline.py --train=data/SNLI/snli_1.0/snli_1.0_train_quebap_v1.json --dev=data/SNLI/snli_1.0/snli_1.0_dev_quebap_v1.json --test=data/SNLI/snli_1.0/snli_1.st_quebap_v1.json --learning_rate=0.001 --l2=0.0001
  ```

#### 4. You can find the list of available models in the models dictionary in the beginning of the training_pipeline.py file
  - We can also use different models. You can look up models that are available in the [training_pipeline.py](../quebap/training_pipeline.py) dictionary:
  ```
    reader_models = {
        'bicond_singlesupport_reader': models.conditional_reader_model,
        'bicond_singlesupport_reader_with_cands': models.conditional_reader_model_with_cands,
        'bilstm_singlesupport_reader_with_cands': models.bilstm_reader_model_with_cands,
        'bilstm_nosupport_reader_with_cands': models.bilstm_nosupport_reader_model_with_cands,
        'boe_support_cands': models.boe_support_cands_reader_model,
        'boe_nosupport_cands': models.boe_nosupport_cands_reader_model,
        'boe_support': models.boe_reader_model,
        'boe_nosupport': models.boenosupport_reader_model,
    }
  ```
  Use the string as an argument for the `model=argument` option, for example `--model=boe_nosupport`

