# How to Install and Run jack

## Install jack and test your installation

The installing procedure currently has three plus one steps:
  1. Install Tensorflow
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

#### 4. [Optional] PyTorch
Jack has preliminary support for pyTorch with an example FastQA implementation in `jack.readers.extractive_qa.torch`
If you want to use pyTorch, please install pyTorch using the instructions on their [web page](http://pytorch.org/)
