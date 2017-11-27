# How to Install and Run jack

## Install jack and test your installation

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
  - Run in the main directory `$ make test` to test the core functionality of jack
  - Run in the main directory `$ make overfit` to test the functionality of the integration between TensorFlow and jack

#### 4. (Optional) Install missing requirements
  - In some cases you might have to manually install some requirements to fix some errors during the `make overfit` procedure. You can install these requirements with `pip3 install -r requirements.txt` in the main directory

## Run an example in jack

There are currently four steps to get an example running with jack:
  1. Download data via script
  2. If your dataset has no specific loader (see `jack/io/load.py`), convert data with preprocessing script into the jack JSON format.
  3. Run bin/jack-train.py with configuration for your dataset and the model that you want
  4. You can find the list of available models in the models dictionary in the beginning of the training_pipeline.py file

#### Available models
  - Different models are available and can be trained using the `model` FLAG of the config. You can look up models that are available in `jack/readers/implementations.py`:
```shell
$ cat jack/readers/implementations.py | grep @ -A 1
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
```
