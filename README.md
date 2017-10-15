# Jack the Reader [![Wercker build badge][wercker_badge]][wercker] [![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/jack-the-reader/Lobby?source=orgpage)
##### A reading comprehension framework.

* All work and no play makes Jack a great frame*work*!
* All work and no play makes Jack a great frame*work*!
* All work and no play makes Jack a great frame*work*!

[wercker_badge]: https://app.wercker.com/status/8ed61192a5b16769a41dc24c30a3bc6a/s/master
[wercker]: https://app.wercker.com/project/byKey/8ed61192a5b16769a41dc24c30a3bc6a
[heres_johnny]: https://upload.wikimedia.org/wikipedia/en/b/bb/The_shining_heres_johnny.jpg

**Jack the Reader** -- or **jack**, for short -- is a framework for building an testing models on a variety of tasks that require *reading comprehension*.

To get started, please see [How to Install and Run][install] and then you may
want to have a look at the [notebooks][notebooks].  Lastly, for a high-level explanation of the ideas and
vision, see [Understanding Jack the Reader][understanding].

[install]: docs/How_to_install_and_run.md
[api]: https://uclmr.github.io/jack/
[notebooks]: notebooks/
[understanding]: docs/Understanding_Jack_the_Reader.md

# Quickstart Examples - Training and Usage of a Question Answering System

To illustrate how jack works, we will show how to train a question answering
model.

### Extractive Question Answering on SQuAD

First, download SQuAD and GloVe embeddings:

```shell
$ cd data/SQuAD/
$ ./download.sh
$ cd ../GloVe/
$ ./download.sh
$ cd ../..
```

Train a [FastQA][fastqa] model

```shell
$ python3 jack/train_reader.py with config='./conf/fastqa.yaml'
```

Note, you can add a flag `tensorboard_folder=.tb/fastqa` to write tensorboard
summaries to a provided path (here `.tb/fastqa`).

A copy of the model is written into the `model_dir` directory after each
training epoch.  These can be loaded using the commands below or see e.g.
[the showcase notebook][showcase].

```python
from jack import readers
from jack.core import QASetting

fastqa_reader = readers.fastqa_reader()
fastqa_reader.load_and_setup("./fastqa_reader")

support = """"It is a replica of the grotto at Lourdes, 
France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. 
At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), 
is a simple, modern stone statue of Mary."""

answers = fastqa_reader([QASetting(
    question="To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
    support=[support]
)])
```

[fastqa]: https://arxiv.org/abs/1703.04816
[showcase]: notebooks/Showcasing_Jack.ipynb

### Recognizing Textual Entailment on SNLI

First, download SNLI

```shell
$ ./data/SNLI/download.sh
```

Lastly, train a [Decomposable Attention Model][dam]

```bash
$ python3 jack/train_reader.py with config=tests/test_conf/dam_test.yaml
```

```python
from jack import readers
from jack.core import QASetting

dam_reader = readers.dam_snli_reader()
dam_reader.load_and_setup("tests/test_results/dam_reader_test")

answers = dam_reader([QASetting(
    question="The boy plays with the ball.",
    support=["The boy plays with the ball."]
)])
```

[dam]: https://arxiv.org/abs/1703.04816

# Support
We are thankful for support from:

<a href="http://mr.cs.ucl.ac.uk/"><img src="http://mr.cs.ucl.ac.uk/images/uclmr_logo_round.png" width="100px"></a>
<a href="http://www.softwarecampus.de/start/df"><img src="https://idw-online.de/de/newsimage?id=186901&size=screen" width="100px"></a>
<a href="http://bloomsbury.ai/"><img src="https://www.dropbox.com/s/7hdb42azs03hbve/logo_text_square.png?raw=1" width="100px"></a>
<a href="https://www.dfki.de/web"><img src="https://www.dfki.de/web/presse/bildmaterial/dfki-logo-e-schrift.jpg" width="100px"></a>
<a href="http://www.pgafamilyfoundation.org"><img src="https://portlandmercado.files.wordpress.com/2013/02/pgaff_pms.jpg" width="100px"></a>
<a href="http://summa-project.eu/"><img src="http://summa-project.eu/wp-content/uploads/2017/04/summalogofinal.png" width="100px"></a>
<a href="http://ec.europa.eu/research/mariecurieactions/funded-projects/career-integration-grants_en"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/European_Commission.svg/2000px-European_Commission.svg.png" width="100px"></a>

# Developer guidelines

- [Comply with the PEP 8 Style Guide][pep8]
- Make sure all your code runs from the top level directory, e.g.:

```shell
$ python3 ./jack/io/SNLI2jtr_v1.py
```

[pep8]: https://www.python.org/dev/peps/pep-0008/
