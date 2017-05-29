[![Wercker build badge][wercker_badge]][wercker]

![Here's Joh^H^H^HJack!][heres_johnny]

* All work and no play makes Jack a great frame*work*!
* All work and no play makes Jack a great frame*work*!
* All work and no play makes Jack a great frame*work*!

[wercker_badge]: https://app.wercker.com/status/8ed61192a5b16769a41dc24c30a3bc6a/s/master
[wercker]: https://app.wercker.com/project/byKey/8ed61192a5b16769a41dc24c30a3bc6a
[heres_johnny]: https://upload.wikimedia.org/wikipedia/en/b/bb/The_shining_heres_johnny.jpg

**Jack the Reader** -- or **jtr**, for short -- is a knowledge base completion
and question answering framework.

To get started, please see [How to Install and Run][install] and then you may
want to have a look at the [API documentation][api] or the
[notebooks][notebooks].  Lastly, for a high-level explanation of the ideas and
vision, see [Understanding Jack the Reader][understanding].

[install]: docs/How_to_install_and_run.md
[api]: https://uclmr.github.io/jtr/
[notebooks]: notebooks/
[understanding]: docs/Understanding_Jack_the_Reader.md

# Quickstart Example: Training a Q&A system #

To illustrate how jtr works, we will show how to train a question answering
model.

First, download SQuAD and GloVe embeddings

```shell
    cd data/SQuAD/
    ./download.sh
    cd ../GloVe/
    ./download.sh
    cd ../..
```

Then, convert SQuAD into the Jack data format

```shell
    python3 jtr/io/SQuAD2jtr.py ./data/SQuAD/data/SQuAD/train-v1.1.json ./data/SQuAD/train.json
    python3 jtr/io/SQuAD2jtr.py ./data/SQuAD/data/SQuAD/dev-v1.1.json ./data/SQuAD/dev.json
```

Lastly, train a [FastQA][fastqa] model

```shell
    python3 jtr/train_reader.py with config='./conf/extractive_qa.yaml'
```

A copy of the model is written into the `model_dir` directory after each
training epoch.  These can be loaded using the commands below or see e.g.
[the showcase notebook][showcase].

```python
    from jtr import readers
    from jtr.core import SharedResources

    svac = SharedResources()

    fastqa_reader = readers.fastqa_reader(svac)
    fastqa_reader.setup_from_file("./fastqa_reader")
```

[fastqa]: https://arxiv.org/abs/1703.04816
[showcase]: notebooks/Showcasing%20Jack.ipynb

# Developer guidelines #

- [Comply with the PEP 8 Style Guide][pep8]
- Make sure all your code runs from the top level directory, e.g.,
    `python3 ./jtr/io/SNLI2jtr_v1.py`

[pep8]: https://www.python.org/dev/peps/pep-0008/
