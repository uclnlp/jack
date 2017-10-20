# How to Install and Run Jack

#### Overview

The installing procedure currently has two steps plus two optional steps:
  1. Install missing dependencies: `pip install -r requirements.txt`
  2. Run `setup.py`: `python setup.py install`
  3. (Optional) Install Tensorflow for your GPU: `pip install tensorflow-gpu`
  4. (Optional) Test your install: `make test`

## Run an example in Jack

To run a model you can invoke a make command, for example `make SNLI` will download the SNLI data and transform it into the Jack-JSON format. To run any model on SNLI is just another command away. To get a list of all commands run `make help`

Here a list of commands that currently work:

##### Downloading and preprocessing datasets:
- `make SNLI`

##### Running models:
- `make SNLI-esim`
