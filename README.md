# jtr -- Jack the Reader
[![wercker status](https://app.wercker.com/status/ebcd272ebfdc8c08c262a000f039bfb8/s/master "wercker status")](https://app.wercker.com/project/byKey/ebcd272ebfdc8c08c262a000f039bfb8)
UCLMR KB and QA system/tools.

![Jack the Reader](https://upload.wikimedia.org/wikipedia/en/b/bb/The_shining_heres_johnny.jpg)

Read [How to Install and Run Jack the Reader](docs/How_to_install_and_run.md) for more detailed information to install and run a first example using Jack the Reader. For a general overview of Jack the Reader and its high-level components see [Understanding Jack the Reader](docs/Understanding_Jack_the_Reader.md). For an overview of the Jack the Reader API see the [API documentation](https://uclmr.github.io/jtr/).


# Preliminaries
Install [word embeddings code](https://github.com/kudkudak/word-embeddings-benchmarks)

[Link](https://docs.google.com/document/d/1AaynDviR26bqofoImEcPxQgBcpvRBIcgLjScn-Hr6xk/edit) to Google Doc for overview/ inventory, etc.

#Rules
- [Comply with PEB 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- Make sure all your code runs from the top level directory, e.g., `$ python3 ./jtr/io/SNLI2jtr_v1.py`

#Training
`$ python3 ./jtr/train.py --train path/to/train --test path/to/test --model your_model_name`

#Tests

See [How to test](./tests/HOW_TO_TEST.md) for information on how to invoke tests and how to add integration tests.

#Completing the loop with MCTest
Let's say you have MCTest downloaded, and at least have two data files, mc160.train.tsv and mc160.train.ans, in your working directory.  We can convert these (together) into jtr format, via stdout, as follows:
```
python3 mctest2jtr.py mc160.train.tsv mc160.train.ans > mc160.jtr.json
```

With the data in the proper format, we can run a baseline model.  Here we use the random baseline (this assumes cwd is jtr/baseline, and we have copied over the above jtr.json file):
```
python3 random_baseline.py mc160.jtr.json > mc160.answers.json
```
The resulting json is just a list of answers, or more precisely, of candidates chosen as answers from the jtr.json file (so "label" fields will still be in these json entries).

How well did we do?  We can score this using the default evaluation script in jtr/eval:
```
python3 eval.py mc160.answers.json mc160.jtr.json
```
This will output a measure of rank accuracy.  The default is to score only the single best answer for each question (k=1), reproducing the standard accuracy used in classification and labeling tasks.

#Validating the Dataset Format
When preparing a new dataset it should stick to a pre-specified json schema.
This can be validated with the `/io/validate.py`. It takes two arguments, the first one specifying the dataset, the second one the schema format. For the above described dataset `mc160.jtr.json` it would work like this (assuming `mc160.jtr.json` would be in the same directory):
```
python3 validate.py mc160.jtr.json dataset_schema.json
```
The schema we use is `/io/dataset_schema.json`.

#Adding Stanford CoreNLP Annotations to Jack the Reader Files
To add annotations to existing Jack the Reader JSON files, use the tools in jtr/preprocess.  This depends on Stanford CoreNLP, so you first must install the python wrapper
```
pip3 install pycorenlp
```
Second, you must download the code and models.  From the top-level jtr directory:
```
cd setup
./download_corenlp.sh
```
Third, start the annotation server:
```
./run_corenlp.sh
```
Cheers to Matko for these scripts.  Finally we are ready to annotate.  Assuming you have a jtr-formatted JSON file in your Jack the Reader directory from the preceding section:
```
python3 preprocess/annotate.py mc160.jtr.json > mc160.jtr.ann.json
```
You now have access to token and sentence offsets, postags for each token, and constituent trees for each sentence.
