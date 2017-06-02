# How to Install and Run jtr

## Install jtr and test your installation

The installing procedure currently has three plus one steps:
  1. Install Tensorflow, HDF5 (hdf5-devel)
  2. Run `setup.py`
  3. Test your install
  4. (Optional) install missing dependencies

#### 1. Install Tensorflow
  - Requirements: Python3, [pip3](http://stackoverflow.com/questions/6587507/how-to-install-pip-with-python-3)
  - GPU requirements: [CUDA 8.0](https://developer.nvidia.com/cuda-downloads) and cuDNN (you need to register with NVIDIA)
  - Follow the TensorFlow [installation guide](https://www.tensorflow.org/get_started/os_setup)

#### 2. Run `setup.py`
  - Run in the main directory: `$ sudo python3 setup.py install`

#### 3. If you run in some problems with missing libraries
  - Run in the main directory `$ make test` to test the core functionality of jtr
  - Run in the main directory `$ make overfit` to test the functionality of the integration between TensorFlow and jtr

#### 4. (Optional) Install missing requirements
  - In some cases you might to manually install some requirements to fix some errors during the `make overfit` procedure. You can install these requirements with `pip3 install -r requirements.txt` in the main directory

## Run an example in jtr

There are currently four steps to get an example running with jtr:
  1. Download data via script
  2. Convert data with preprocessing script into the jtr JSON format
  3. Run training_pipeline.py with parameters for your dataset and the model that you want
  4. You can find the list of available models in the models dictionary in the beginning of the training_pipeline.py file

#### 1. Download data via script
  - Have a look to jtr/data to get an overview over the currently available dataset
  - Download the data by running the download.sh script (not all dataset currently have such a script), for example for SNLI cd into its directory and run `sh download.sh`

#### 2. Convert data with preprocessing script into the jtr JSON format
  - Go to the main jtr directory and execute the preprocessing script located in jtr/load for example for SNLI you can run `python3 jtr/load/SNLI2jtr_v1.py` the data will be converted and rewritten into the jtr/data directory

#### 3. Run training_pipeline.py with parameters for your dataset and the model that you want
  - The main point of entry to run a model with jtr is `jtr/train_reader.py` we need to pass some parameters for our data into the script along with the model that we want to run. By default we run a pair of conditional bidirectional LSTM readers that make use of a single supporting information (input: Question; support: Supporting document which should be used to answer that question)

#### 4. You can find the list of available models in the models dictionary in the beginning of the training_pipeline.py file
  - We can also use different models. You can look up models that are available in `jtr/readers.py`:
```shell
$ cat jtr/readers.py | grep @ -A 1
@__mcqa_reader
def example_reader(shared_resources: SharedResources):
--
@__kbp_reader
def modelf_reader(shared_resources: SharedResources):
--
@__kbp_reader
def distmult_reader(shared_resources: SharedResources):
--
@__kbp_reader
def complex_reader(shared_resources: SharedResources):
--
@__kbp_reader
def transe_reader(shared_resources: SharedResources):
--
@__xqa_reader
def fastqa_reader(shared_resources: SharedResources):
--
@__xqa_reader
def cbow_xqa_reader(shared_resources: SharedResources):
--
@__mcqa_reader
def cbilstm_snli_reader(shared_resources: SharedResources):
--
@__mcqa_reader
def dam_snli_reader(shared_resources: SharedResources):
--
@__mcqa_reader
def esim_snli_reader(shared_resources: SharedResources):
--
@__mcqa_reader
def snli_reader(shared_resources: SharedResources):
```
