# Jack the Reader [![Wercker build badge][wercker_badge]][wercker] [![codecov](https://codecov.io/gh/uclmr/jack/branch/master/graph/badge.svg?token=jbZrj9oSmi)](https://codecov.io/gh/uclmr/jack) [![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/jack-the-reader/Lobby?source=orgpage) [![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/uclmr/jack/blob/master/LICENSE)

##### A Machine Reading Comprehension framework.

* All work and no play makes Jack a great frame*work*!
* All work and no play makes Jack a great frame*work*!
* All work and no play makes Jack a great frame*work*!

[wercker_badge]: https://app.wercker.com/status/8ed61192a5b16769a41dc24c30a3bc6a/s/master
[wercker]: https://app.wercker.com/project/byKey/8ed61192a5b16769a41dc24c30a3bc6a
[heres_johnny]: https://upload.wikimedia.org/wikipedia/en/b/bb/The_shining_heres_johnny.jpg

**Jack the Reader** - or **jack**, for short - is a framework for building and using models on a variety of tasks that require *reading comprehension*. For more informations about the overall architecture, we refer to [Jack the Reader – A Machine Reading Framework](https://arxiv.org/abs/1806.08727) (ACL 2018).

## Installation
To install Jack, install requirements and [TensorFlow](http://tensorflow.org/). In case you want to use PyTorch for writing models, please install [PyTorch](http://pytorch.org/) as well.

## Supported ML Backends

We currently support [TensorFlow](http://tensorflow.org/) and [PyTorch](http://pytorch.org/).
Readers can be implemented using both. Input and output modules (i.e., pre- and post-processing) are independent of the
ML backend and can thus be reused for model modules that either backend.
Though most models are implemented in TensorFlow by reusing the cumbersome pre- and post-processing it is easy to
quickly build new readers in PyTorch as well.

## Pre-trained Models

Find pre-trained models [here](https://www.dropbox.com/sh/vnmc9pq4yrgl1sr/AAD7HVhGdpof2IgIifSZ6PEUa?dl=0).

## Code Structure

* `jack.core` - core abstractions used
* `jack.readers` - implementations of models
* `jack.eval` - task evaluation code
* `jack.util` - utility code that is used throughout the framework, including shared ML code
* `jack.io` - IO related code, including loading and dataset conversion scripts


## Projects

* [Integration of Knowledge into neural NLU systems](/projects/knowledge_integration)

## Quickstart

### Coding Tutorials - Notebooks & CLI
We provide ipython notebooks with tutorials on Jack. For the quickest start, you can begin [here][quickstart]. If you're interested in training a model yourself from code, see [this tutorial][model_training] (we recommend the command-line, see below), and if you'd like to implement a new model yourself, [this notebook][implementation] gives you a tutorial that explains this process in more detail.

There is documentation on our [command-line interface][cli] for actually **training and evaluating models**.
For a high-level explanation of the ideas and vision, see [Understanding Jack the Reader][understanding].

[quickstart]: notebooks/quick_start.ipynb
[model_training]: notebooks/model_training.ipynb
[implementation]: notebooks/model_implementation.ipynb
[install]: docs/How_to_install_and_run.md
[api]: https://uclmr.github.io/jack/
[notebooks]: notebooks/
[understanding]: docs/Understanding_Jack_the_Reader.md
[cli]: docs/CLI.md

### Command-line Training and Usage of a QA System
To illustrate how jack works, we will show how to train a question answering
model using our [command-line interface][cli] which is analoguous for other tasks (browse [conf/](./conf/) for existing task-dataset configurations).
It is probably best to setup a [virtual environment](https://docs.python.org/3/library/venv.html) to avoid
clashes with system wide python library versions.

First, install the framework:

```bash
$ python3 -m pip install -e .[tf]
```

Then, download the SQuAD dataset, and the GloVe word embeddings:

```bash
$ ./data/SQuAD/download.sh
$ ./data/GloVe/download.sh
```

Train a [FastQA][fastqa] model:

```bash
$ python3 bin/jack-train.py with train='data/SQuAD/train-v1.1.json' dev='data/SQuAD/dev-v1.1.json' reader='fastqa_reader' \
> repr_dim=300 dropout=0.5 batch_size=64 seed=1337 loader='squad' save_dir='./fastqa_reader' epochs=20 \
> with_char_embeddings=True embedding_format='memory_map_dir' embedding_file='data/GloVe/glove.840B.300d.memory_map_dir' vocab_from_embeddings=True
```

or shorter, using our prepared config:

```bash
$ python3 bin/jack-train.py with config='./conf/qa/squad/fastqa.yaml'
```

A copy of the model is written into the `save_dir` directory after each
training epoch when performance improves. These can be loaded using the commands below or see e.g.
[quickstart].

You want to train another model? No problem, we have a fairly modular QAModel implementation which allows you to stick
together your own model. There are examples in `conf/qa/squad/` (e.g., `bidaf.yaml` or our own creation `jack_qa.yaml`).
These models are defined solely in the configs, i.e., there is not implementation in code.
This is possible through our `ModularQAModel`.

If all of that is too cumbersome for you and you just want to play, why not downloading a pretrained model:

```bash
$ # we still need GloVe in memory mapped format, ignore the next 2 commands if already downloaded and transformed
$ data/GloVe/download.sh
$ wget -O fastqa.zip https://www.dropbox.com/s/qb796uljoqj0lvo/fastqa.zip?dl=1
$ unzip fastqa.zip && mv fastqa fastqa_reader
```

```python
from jack import readers
from jack.core import QASetting

fastqa_reader = readers.reader_from_file("./fastqa_reader")

support = """"It is a replica of the grotto at Lourdes,
France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858.
At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome),
is a simple, modern stone statue of Mary."""

answers = fastqa_reader([QASetting(
    question="To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
    support=[support]
)])

print(answers[0][0].text)
```
[fastqa]: https://arxiv.org/abs/1703.04816
[tf_summaries]: https://www.tensorflow.org/get_started/summaries_and_tensorboard
[quick_start]: notebooks/quick_start.ipynb
[cli]: docs/CLI.md

## Support
We are thankful for support from:

<a href="http://mr.cs.ucl.ac.uk/"><img src="http://mr.cs.ucl.ac.uk/images/uclmr_logo_round.png" width="100px"></a>
<a href="http://www.softwarecampus.de/start/df"><img src="https://idw-online.de/de/newsimage?id=186901&size=screen" width="100px"></a>
<a href="http://ec.europa.eu/research/mariecurieactions/funded-projects/career-integration-grants_en"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/European_Commission.svg/2000px-European_Commission.svg.png" width="100px"></a>

<a href="http://bloomsbury.ai/"><img src="https://www.dropbox.com/s/7hdb42azs03hbve/logo_text_square.png?raw=1" width="100px"></a>
<a href="https://www.dfki.de/web"><img src="https://www.dfki.de/web/presse/bildmaterial/dfki-logo-e-schrift.jpg" width="100px"></a>
<a href="http://www.pgafamilyfoundation.org"><img src="https://portlandmercado.files.wordpress.com/2013/02/pgaff_pms.jpg" width="100px"></a>
<a href="http://summa-project.eu/"><img src="http://summa-project.eu/wp-content/uploads/2017/04/summalogofinal.png" width="100px"></a>

## Developer guidelines

- [Comply with the PEP 8 Style Guide][pep8]
- Make sure all your code runs from the top level directory, e.g.:

```shell
$ pwd
/home/pasquale/workspace/jack
$ python3 bin/jack-train.py [..]
```

[pep8]: https://www.python.org/dev/peps/pep-0008/

## Citing

```
@InProceedings{weissenborn2018jack,
author    = {Dirk Weissenborn, Pasquale Minervini, Tim Dettmers, Isabelle Augenstein, Johannes Welbl, Tim Rocktäschel, Matko Bošnjak, Jeff Mitchell, Thomas Demeester, Pontus Stenetorp, Sebastian Riedel},
title     = {{Jack the Reader – A Machine Reading Framework}},
booktitle = {{Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL) System Demonstrations}},
Month     = {July},
year      = {2018},
url       = {https://arxiv.org/abs/1806.08727}
}
```
