# Jack the Reader [![Wercker build badge][wercker_badge]][wercker] [![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/jack-the-reader/Lobby?source=orgpage)
##### A knowledge base completion and question answering framework.

* All work and no play makes Jack a great frame*work*!
* All work and no play makes Jack a great frame*work*!
* All work and no play makes Jack a great frame*work*!

[wercker_badge]: https://app.wercker.com/status/8ed61192a5b16769a41dc24c30a3bc6a/s/master
[wercker]: https://app.wercker.com/project/byKey/8ed61192a5b16769a41dc24c30a3bc6a
[heres_johnny]: https://upload.wikimedia.org/wikipedia/en/b/bb/The_shining_heres_johnny.jpg

**Jack the Reader** -- or **jack**, for short -- is a knowledge base completion
and question answering framework.

To get started, please see [How to Install and Run][install] and then you may
want to have a look at the [API documentation][api] or the
[notebooks][notebooks].  Lastly, for a high-level explanation of the ideas and
vision, see [Understanding Jack the Reader][understanding].

[install]: docs/How_to_install_and_run.md
[api]: https://uclmr.github.io/jack/
[notebooks]: notebooks/
[understanding]: docs/Understanding_Jack_the_Reader.md

# Quickstart Example: Training a Q&A system #

To illustrate how jack works, we will show how to train a question answering
model.

First, download SQuAD and GloVe embeddings

```shell
$ cd data/SQuAD/
$ ./download.sh
$ cd ../GloVe/
$ ./download.sh
$ cd ../..
```

Then, convert SQuAD into the Jack data format

```shell
$ python3 jack/io/SQuAD2jtr.py ./data/SQuAD/train-v1.1.json ./data/SQuAD/train.json
$ python3 jack/io/SQuAD2jtr.py ./data/SQuAD/dev-v1.1.json ./data/SQuAD/dev.json
```

Lastly, train a [FastQA][fastqa] model

```shell
$ python3 jack/train_reader.py with config='./conf/fastqa.yaml'
```

Note, you can add a flag `tensorboard_folder=.tb/fastqa` to write tensorboard
summaries to a provided path (here ".tb/fastqa").

A copy of the model is written into the `model_dir` directory after each
training epoch.  These can be loaded using the commands below or see e.g.
[the showcase notebook][showcase].

```python
from jack import readers

fastqa_reader = readers.fastqa_reader()
fastqa_reader.load_and_setup("./fastqa_reader")
```

[fastqa]: https://arxiv.org/abs/1703.04816
[showcase]: notebooks/Showcasing_Jack.ipynb

# Developer guidelines #

- [Comply with the PEP 8 Style Guide][pep8]
- Make sure all your code runs from the top level directory, e.g.:

```shell
$ python3 ./jack/io/SNLI2jtr_v1.py
```

[pep8]: https://www.python.org/dev/peps/pep-0008/
