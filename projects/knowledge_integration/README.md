## About

This project is an open source implementation for the paper:

*Dynamic Integration of Background Knowledge in Neural NLU Systems.* </br>
Dirk Weissenborn, Tomáš Kočiský, Chris Dyer.</br>
[ArXiv](https://arxiv.org/abs/1706.02596)

It is fully integrated into JACK and thus training/evaluation scripts, notebooks, etc., work with these implementations.

## Setup
At first you need to download and setup the necessary resources. Make sure you installed JACK and your 
current working directory is the project directory.

### ConceptNet

```bash
$ export DATA_DIR=data/knowledge_integration
$ mkdir $DATA_DIR
$ wget -P $DATA_DIR https://s3.amazonaws.com/conceptnet/precomputed-data/2016/assertions/conceptnet-assertions-5.5.0.csv.gz
$ export PYTHONPATH=$PYTHONPATH:.
$ python3 projects/knowledge_integration/scripts/extract_conceptnet.py $DATA_DIR/conceptnet-assertions-5.5.0.csv.gz $DATA_DIR/knowledge_store
```

### Wikipedia Abstracts

```bash
$ wget -P $DATA_DIR http://downloads.dbpedia.org/2016-10/core-i18n/en/short_abstracts_en.ttl.bz2
$ wget -P $DATA_DIR http://downloads.dbpedia.org/2016-10/core-i18n/en/anchor_text_en.ttl.bz2
$ wget -P $DATA_DIR http://downloads.dbpedia.org/2016-10/core-i18n/en/disambiguations_en.ttl.bz2
$ wget -P $DATA_DIR http://downloads.dbpedia.org/2016-10/core-i18n/en/transitive_redirects_en.ttl.bz2
$ python3 projects/knowledge_integration/scripts/extract_wikipedia_short_abstract.py \
        --assertion_store_path $DATA_DIR/assertion_store \
        --short_abstracts $DATA_DIR/short_abstracts_en.ttl.bz2 \
        --anchor_texts $DATA_DIR/anchor_text_en.ttl.bz2 \
        --disambiguations $DATA_DIR/disambiguations_en.ttl.bz2 \
        --transitive_redirects $DATA_DIR/transitive_redirects_en.ttl.bz2
```

### Pre-trained Word Embeddings

Download pre-trained word embeddings and training data (see [data/](/data/)) if not already done:

```
$ ./data/GloVe/download.sh
```


## Models & Training

Find model configurations in [conf/](/projects/knowledge_integration/conf).

### QA

E.g., SQuAD:
```
$ ./data/SQuAD/download.sh
$ # with conceptnet integration:
$ python3 bin/jack-train.py with config=projects/knowledge_integration/conf/qa/squad/bilstm_assertion.yaml
$ # with wikipedia abstract integration run:
$ python3 bin/jack-train.py with config=projects/knowledge_integration/conf/qa/squad/bilstm_assertion_definition.yaml
```
Training on TriviaQA is similar with the prepared configs.

### NLI

E.g., SNLI:
```
$ ./data/SNLI/download.sh
$ # with conceptnet integration:
$ python3 bin/jack-train.py with config=projects/knowledge_integration/conf/nli/snli/cbilstm_assertion.yaml
```

Training on MultiNLI is similar with the prepared configs.
