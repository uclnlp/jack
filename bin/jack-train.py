#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import os.path as path
import sys
from time import time

from sacred import Experiment
from sacred.arg_parser import parse_args
from sacred.observers import SqlObserver

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


def fetch_parents(current_path, parents=[]):
    tmp_ex = Experiment('jack')
    tmp_ex.add_config(current_path)
    if "parent_config" in tmp_ex.configurations[0]._conf:
        return fetch_parents(tmp_ex.configurations[0]._conf["parent_config"], [current_path] + parents)
    else:
        return [current_path] + parents

configs = fetch_parents(path)
logger.info("Loading {}".format(configs))
ex = Experiment('jack')
for path in configs:
    ex.add_config(path)


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
def main(batch_size,
         clip_value,
         config,
         loader,
         debug,
         debug_examples,
         dev,
         embedding_file,
         embedding_format,
         experiments_db,
         epochs,
         l2,
         optimizer,
         learning_rate,
         learning_rate_decay,
         log_interval,
         validation_interval,
         model,
         model_dir,
         seed,
         tensorboard_folder,
         test,
         train,
         vocab_from_embeddings,
         write_metrics_to):
    logger.info("TRAINING")

    if experiments_db is not None:
        ex.observers.append(SqlObserver.create('sqlite:///%s' % experiments_db))

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
        train_data = loaders[loader](train)
        dev_data = loaders[loader](dev)
        test_data = loaders[loader](test) if test else None

        logger.info('loaded train/dev/test data')
        if embedding_file is not None and embedding_format is not None:
            embeddings = load_embeddings(embedding_file, embedding_format)
            logger.info('loaded pre-trained embeddings ({})'.format(embedding_file))
            ex.current_run.config["repr_dim_input"] = embeddings.lookup[0].shape[0]
        else:
            embeddings = Embeddings(None, None)

    emb = embeddings

    vocab = Vocab(emb=emb, init_from_embeddings=vocab_from_embeddings)

    # build JTReader
    checkpoint()
    parsed_config = ex.current_run.config
    ex.run('print_config', config_updates=parsed_config)

    # name defaults to name of the model
    if 'name' not in parsed_config or parsed_config['name'] is None:
        parsed_config['name'] = model

    shared_resources = SharedResources(vocab, parsed_config)
    reader = readers.readers[model](shared_resources)

    checkpoint()

    configuration = {
        'seed': seed,
        'clip_value': clip_value,
        'batch_size': batch_size,
        'epochs': epochs,
        'l2': l2,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'learning_rate_decay': learning_rate_decay,
        'log_interval': log_interval,
        'validation_interval': validation_interval,
        'tensorboard_folder': tensorboard_folder,
        'model': model,
        'model_dir': model_dir,
        'write_metrics_to': write_metrics_to
    }

    jtrain(reader, train_data, test_data, dev_data, configuration, debug=debug)
