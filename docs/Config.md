# Configs and Command Line Arguments for Jack

Jack uses [sacred](http://sacred.readthedocs.io/en/latest/) for management of config files and parsing of command files.


Configuration files are stored in `../conf`: see `./conf/jack.yaml` for a default configuration file.

Configuration files use [YAML](http://www.yaml.org/start.html) as markup language to define hyperparameters and other options for training and running Jack models.

Configuration files can inherit default configurations by setting a `parent_config`.
For instance, `./conf/fastqa.yaml` inherits and overrides parameters of `./conf/jack.yaml`, containing the default configuration for all models.

## Selecting a Config

Paths to configs are passed in the command line.
For example, the following runs Jack with the `conf/fastqa.yaml` config.

```shell
$ python3 bin/jack-train.py with config=conf/fastqa.yaml
```

If no `config` is specified, the default configuration `conf/jack.yaml` is used.

## Command Line Arguments

Config parameters can be overriden by command line arguments. For instance, the following overrides the l2 regularization strength to `0.001`:

```shell
$ python3 jtr/train_reader.py with config='conf/jack_more_specific.yaml' l2=0.001
```

## Printing Config

Passing `print_config` to the reader shows the config that is used.

```shell
$ python3 bin/jack-train.py print_config with config=conf/fastqa.yaml l2=0.001
```

Note that configuration parameters that are overriden using the command line are highlighted in blue.

## Saving Config

Passing `save_config` saves the current configuration as a YAML configuration file named `config.json` in the working directory.

```shell
$ python3 bin/jack-train.py save_config with config=conf/fastqa.yaml l2=0.001
```

## Notes

Examples in this file require Downloading the training data and GloVe embeddings

```shell
$ cd data/SQuAD/
$ ./download.sh
$ cd ../GloVe/
$ ./download.sh
```
