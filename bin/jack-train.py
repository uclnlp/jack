#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import os.path as path
import shutil
import sys
import tempfile
import uuid
from time import time

from sacred import Experiment
from sacred.arg_parser import parse_args

from jack import readers
from jack import train as jtrain
from jack.core.shared_resources import SharedResources
from jack.io.embeddings.embeddings import load_embeddings, Embeddings
from jack.io.load import loaders
from jack.util.vocab import Vocab

logger = logging.getLogger(os.path.basename(sys.argv[0]))

parsed_args = dict([x.split("=") for x in parse_args(sys.argv)["UPDATE"]])
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


class Duration(object):
    def __init__(self):
        self.t0 = time()
        self.t = time()

    def __call__(self):
        logger.info('Time since last checkpoint : {0:.2g}min'.format((time() - self.t) / 60.))
        self.t = time()


checkpoint = Duration()

logging.basicConfig(level=logging.INFO)


@ex.automain
def main(config,
         loader,
         debug,
         debug_examples,
         embedding_file,
         embedding_format,
         reader,
         train,
         num_train_examples,
         dev,
         num_dev_examples,
         test,
         vocab_from_embeddings):
    logger.info("TRAINING")

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
            ex.current_run.config["repr_dim_input"] = 50
        else:
            embeddings = Embeddings(None, None)
    else:
        train_data = loaders[loader](train, num_train_examples)
        dev_data = loaders[loader](dev, num_dev_examples)
        test_data = loaders[loader](test) if test else None

        logger.info('loaded train/dev/test data')
        if embedding_file is not None and embedding_format is not None:
            embeddings = load_embeddings(embedding_file, embedding_format)
            logger.info('loaded pre-trained embeddings ({})'.format(embedding_file))
            ex.current_run.config["repr_dim_input"] = embeddings.lookup[0].shape[0]
        else:
            embeddings = None
            if ex.current_run.config["vocab_from_embeddings"]:
                raise RuntimeError("If you want to create vocab from embeddings, embeddings have to be provided")

    vocab = Vocab(emb=embeddings, init_from_embeddings=vocab_from_embeddings)

    # build JTReader
    checkpoint()
    parsed_config = ex.current_run.config
    ex.run('print_config', config_updates=parsed_config)

    # name defaults to name of the model
    if 'name' not in parsed_config or parsed_config['name'] is None:
        parsed_config['name'] = reader

    shared_resources = SharedResources(vocab, parsed_config)
    jtreader = readers.readers[reader](shared_resources)

    checkpoint()

    try:
        jtrain(jtreader, train_data, test_data, dev_data, parsed_config, debug=debug)
    finally:  # clean up temporary dir
        if os.path.exists(jack_temp):
            shutil.rmtree(jack_temp)
