# jtr -- Jack the Reader ///////////
[![wercker status](https://app.wercker.com/status/8ed61192a5b16769a41dc24c30a3bc6a/s/master "wercker status")](https://app.wercker.com/project/byKey/8ed61192a5b16769a41dc24c30a3bc6a)
UCLMR KB and QA system/tools.

![Jack the Reader](https://upload.wikimedia.org/wikipedia/en/b/bb/The_shining_heres_johnny.jpg)

All work and no play makes Jack a great framework!

All work and no play makes Jack a great framework!

All work and no play makes Jack a great framework!

Read [How to Install and Run Jack the Reader](docs/How_to_install_and_run.md) for more detailed information to install and run a first example using Jack the Reader. For a general overview of Jack the Reader and its high-level components see [Understanding Jack the Reader](docs/Understanding_Jack_the_Reader.md). For an overview of the Jack the Reader API see the [API documentation](https://uclmr.github.io/jtr/).

# Rules
- [Comply with PEB 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- Make sure all your code runs from the top level directory, e.g., `$ python3 ./jtr/io/SNLI2jtr_v1.py`


## Quickstart Example: Train Q&A
Download SQuAD and GloVe embeddings
```shell
cd data/SQuAD/
./download.sh
cd ../GloVe/
./download.sh
```

Convert SQuAD into Jack format
```shell
python3 jtr/convert/SQuAD2jtr.py ./data/SQuAD/data/SQuAD/train-v1.1.json ./data/SQuAD/train.json
python3 jtr/convert/SQuAD2jtr.py ./data/SQuAD/data/SQuAD/dev-v1.1.json ./data/SQuAD/dev.json
```

Train FastQA
```shell
python3 jtr/jack/train/train_sacred_reader.py with config='./conf/extractive_qa.yaml'
```
