#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import os.path as path
import shutil
import sys
import tempfile
import uuid

from sacred import Experiment
from sacred.arg_parser import parse_args

from jack import readers
from jack import train as jtrain
from jack.core.shared_resources import SharedResources
from jack.io.embeddings.embeddings import load_embeddings
from jack.io.load import loaders
from jack.util.vocab import Vocab

# register knowledge integration models

logger = logging.getLogger(os.path.basename(sys.argv[0]))

parsed_args = dict(x.split("=") for x in parse_args(sys.argv)["UPDATE"])
if "config" in parsed_args:
    path = parsed_args["config"]
else:
    path = "./conf/jack.yaml"


def fetch_parents(current_path):
    tmp_ex = Experiment('jack')
    if not isinstance(current_path, list):
        current_path = [current_path]
    all_paths = list(current_path)
    for p in current_path:
        tmp_ex.add_config(p)
        if "parent_config" in tmp_ex.configurations[-1]._conf:
            all_paths = fetch_parents(tmp_ex.configurations[-1]._conf["parent_config"]) + all_paths
    return all_paths


configs = fetch_parents(path)
logger.info("Loading {}".format(configs))
ex = Experiment('jack')
for c_path in configs:
    ex.add_config(c_path)

logging.basicConfig(level=logging.INFO)


@ex.automain
def run(loader,
        debug,
        debug_examples,
        embedding_file,
        embedding_format,
        repr_dim_task_embedding,
        reader,
        train,
        num_train_examples,
        dev,
        num_dev_examples,
        test,
        vocab_from_embeddings,
        **kwargs):
    logger.info("TRAINING")

    # build JTReader
    parsed_config = ex.current_run.config
    ex.run('print_config', config_updates=parsed_config)

    if 'JACK_TEMP' not in os.environ:
        jack_temp = os.path.join(tempfile.gettempdir(), 'jack', str(uuid.uuid4()))
        os.environ['JACK_TEMP'] = jack_temp
        logger.info("JACK_TEMP not set, setting it to %s. Might be used for caching." % jack_temp)
    else:
        jack_temp = os.environ['JACK_TEMP']
    if not os.path.exists(jack_temp):
        os.makedirs(jack_temp)

    if debug:
        train_data = loaders[loader](train, debug_examples)

        logger.info('loaded {} samples as debug train/dev/test dataset '.format(debug_examples))

        dev_data = train_data
        test_data = train_data

        if embedding_file is not None and embedding_format is not None:
            emb_file = 'glove.6B.50d.txt'
            embeddings = load_embeddings(path.join('data', 'GloVe', emb_file), 'glove')
            logger.info('loaded pre-trained embeddings ({})'.format(emb_file))
        else:
            embeddings = None
    else:
        train_data = loaders[loader](train, num_train_examples)
        dev_data = loaders[loader](dev, num_dev_examples)
        test_data = loaders[loader](test) if test else None

        logger.info('loaded train/dev/test data')
        if embedding_file is not None and embedding_format is not None:
            embeddings = load_embeddings(embedding_file, embedding_format)
            logger.info('loaded pre-trained embeddings ({})'.format(embedding_file))
        else:
            embeddings = None
            if vocab_from_embeddings:
                raise ValueError("If you want to create vocab from embeddings, embeddings have to be provided")

    vocab = Vocab(vocab=embeddings.vocabulary if vocab_from_embeddings and embeddings is not None else None)

    if repr_dim_task_embedding < 1 and embeddings is None:
        raise ValueError("Either provide pre-trained embeddings or set repr_dim_task_embedding > 0.")


    # name defaults to name of the model
    if 'name' not in parsed_config or parsed_config['name'] is None:
        parsed_config['name'] = reader

    shared_resources = SharedResources(vocab, parsed_config, embeddings)
    jtreader = readers.readers[reader](shared_resources)

    try:
        jtrain(jtreader, train_data, test_data, dev_data, parsed_config, debug=debug)
    finally:  # clean up temporary dir
        if os.path.exists(jack_temp):
            shutil.rmtree(jack_temp)
