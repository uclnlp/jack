# Training in Jack

Training on jack is realized through the `bin/jack-train.py` CLI.

## Configs and Command Line Arguments for Jack

Jack uses [sacred](http://sacred.readthedocs.io/en/latest/) for management of config files and parsing of command files.

Configuration files are stored in `./conf`: see `./conf/jack.yaml` for a default configuration file and the list of 
flags that can be used. See model configurations in subdirectories for examples.

Configuration files use [YAML](http://www.yaml.org/start.html) as markup language to define hyperparameters and other options for training and running Jack models.

Configuration files can inherit default configurations by setting a `parent_config`.
For instance, `./conf/qa/squad/fastqa.yaml` inherits and overrides parameters of `./conf/qa/fastqa.yaml` which overrides
`./conf/jack.yaml`, containing the default configuration for all models. Note, that you can also introduce new configuration parameters in a child config if you need it for your model. There is no need to first introduce it in the parent config.

### Selecting a Config

Paths to configs are passed in the command line.
For example, the following runs Jack with the `conf/qa/squad/fastqa.yaml` config.

```shell
$ python3 bin/jack-train.py with config=conf/qa/squad/fastqa.yaml
```

If no `config` is specified, the default configuration `conf/jack.yaml` is used.

### Command Line Arguments

Config parameters can be overridden by command line arguments. For instance, the following overrides the l2 regularization strength to `0.001`:

```shell
$ python3 bin/jack-train.py with config='conf/jack.yaml' l2=0.001
$ python3 bin/jack-train.py with l2=0.001  # equivalent
```

### Printing Config

Passing `print_config` to the reader shows the config that is used.

```shell
$ python3 bin/jack-train.py print_config with config=conf/qa/squad/fastqa.yaml l2=0.001
```

Note that configuration parameters that are overridden using the command line are highlighted in blue.

### Saving Config

Passing `save_config` saves the current configuration as a YAML configuration file named `config.json` in the working directory.

```shell
$ python3 bin/jack-train.py save_config with config=conf/fastqa.yaml l2=0.001
```

## Important Training Flags

* `learning_rate`: learning rate used in the optimizer (1e-3 by default)
* `optimizer`: 'adam' (default), 'gd' (simple gradient descent), 'adagrad', 'adadelta', 'rmsprop'
* `train`, `dev`, `test`: dataset files. `test` is optional
* `loader`: different file formats that we support: 'jack' (default), 'squad', 'snli'. There are converters to 'jack'
format for datasets that are not natively supported.
* `learning_rate_decay`: multiplicative learning rate decay whenever dev set performance drops (default 1.0, i.e., no decay).
* `save_dir`: where to save the model when it improves on dev set (default: 'saved_reader'). 
* `load_dir`: load pretrained model before training (partially, i.e., the pre-trained model can contain only a subset of the total parameters)
* `validation_interval`: number of mini-batches between validation on dev set (-1 by default, that is, after each epoch)
* `batch_size`: size of mini-batches to train on

## Notes

There are many predefined configs that you can use for training out of the box after downloading the required resources. We also provide several pretrained models that you can find within the dedicated task [documentation](./tasks). Examples in this file require downloading the training data and GloVe embeddings

```shell
$ data/SQuAD/download.sh
$ data/GloVe/download.sh
```

# Evaluation

After training we usually want to properly evaluate our model on test data. This can be done through the
`bin/jack-eval.py` CLI. Flags are:

* `--load_dir`: path to your saved reader
* `--dataset`: path to dataset to evaluate on
* `--loader`: loader for the dataset (default: 'jack')
* `--batch_size`: batch_size to use when processing the dataset
* `--max_examples`: can be specified to only evaluate on the initial examples in the dataset
* `--overwrite`: sometimes you might want to override some configuration of your reader during evaluation.
You can write a json string that will override the stored reader configuration. You can of course change the reader 
configuration also manually in the `save_dir` itself. 
