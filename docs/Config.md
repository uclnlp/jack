# Configs and Command Line Arguments for Jack

Jack uses [sacred](http://sacred.readthedocs.io/en/latest/) for management of config files and parsing of command files.


Configuration files are stored in `../conf`: see `./conf/jack.yaml` for a default configuration file.

Configuration files use [YAML](http://www.yaml.org/start.html) as markup language to define hyperparameters and other options for training and running Jack models.

Configuration files can inherit default configurations by setting a `parent_config`.
For instance, `./conf/fastqa.yaml` inherits and overrides parameters of `./conf/jack.yaml`, containing the default configuration for all models.

## Selecting a Config
Paths to configs are passed in the command line. For example, the following runs Jack with the `jack_more_specific.yaml` config.
```shell
$ python3 jtr/train_reader.py with config=‘../conf/jack_more_specific.yaml'
```
If no `config` is specified, the default configuration `./conf/jack.yaml` is used.

## Command Line Arguments
Config parameters can be overriden by command line arguments. For instance, the following overrides the l2 regularization strength to `0.001`:
```shell
$ python3 jtr/train_reader.py with config='./conf/jack_more_specific.yaml' l2=0.001
```

## Printing Config
Passing `print_config` to the reader shows the config that is used.
```shell
$ python3 jtr/train_reader.py print_config with config='./conf/jack_more_specific.yaml' l2=0.001
```
Note that configuration parameters that are overriden using the command line are highlighted in blue.

## Saving Config
Passing `save_config` saves the used config file as YAML in the working directory.
```shell
$ python3 jtr/train_reader.py save_config with config='./conf/jack_more_specific.yaml' l2=0.001
```

## Example: Extractive Q&A
Download data and GloVe embeddings
```shell
$ cd data/SQuAD/
$ ./download.sh
$ cd ../GloVe/
$ ./download.sh
```

Convert SQuAD into Jack format
```shell
python3 jtr/convert/SQuAD2jtr.py ./data/SQuAD/data/SQuAD/train-v1.1.json ./data/SQuAD/train.json
python3 jtr/convert/SQuAD2jtr.py ./data/SQuAD/data/SQuAD/dev-v1.1.json ./data/SQuAD/dev.json
```

Run FastQA
```shell
python3 jtr/jack/train/train_sacred_reader.py with config='./conf/extractive_qa.yaml'
```