# -*- coding: utf-8 -*-

import logging
import os
import os.path as path
import random
import sys

import tensorflow as tf
from sacred import Experiment
from sacred.arg_parser import parse_args
from sacred.observers import SqlObserver

from jtr import readers
from jtr.core import SharedResources
from jtr.data_structures import load_labelled_data, load_labelled_data_stream
from jtr.input_output.embeddings.embeddings import load_embeddings, Embeddings
from jtr.input_output.stream_processors import dataset2stream_processor
from jtr.train_tools import train_reader
from jtr.util.vocab import Vocab

logger = logging.getLogger(os.path.basename(sys.argv[0]))

parsed_args = dict([x.split("=") for x in parse_args(sys.argv)["UPDATE"]])
if "config" in parsed_args:
    path = parsed_args["config"]
else:
    path = "./conf/jack.yaml"


def fetch_parents(current_path, parents=[]):
    tmp_ex = Experiment('jack')
    tmp_ex.add_config(current_path)
    tmp_ex.run("print_config")
    if tmp_ex.current_run is not None and "parent_config" in tmp_ex.current_run.config:
        return fetch_parents(tmp_ex.current_run.config["parent_config"], [current_path] + parents)
    else:
        return [current_path] + parents


configs = fetch_parents(path)
logger.info("Loading {}".format(configs))
ex = Experiment('jack')
for path in configs:
    ex.add_config(path)

logger.info(ex.current_run)

logging.basicConfig(level=logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # print only TF errors


@ex.automain
def main(batch_size,
         clip_value,
         debug,
         debug_examples,
         dev,
         embedding_file,
         embedding_format,
         experiments_db,
         epochs,
         l2,
         learning_rate,
         learning_rate_decay,
         log_interval,
         validation_interval,
         model,
         model_dir,
         pretrain,
         seed,
         tensorboard_folder,
         test,
         train,
         write_metrics_to,
         use_streaming,
         dataset_name):
    logger.info("TRAINING")

    if experiments_db is not None:
        ex.observers.append(SqlObserver.create('sqlite:///%s' % experiments_db))

    # make everything deterministic
    random.seed(seed)
    tf.set_random_seed(seed)

    if clip_value != 0.0:
        clip_value = - abs(clip_value), abs(clip_value)

    if debug:
        if not use_streaming:
            train_data = load_labelled_data(train, debug_examples)
        else:
            train_data = load_labelled_data_stream(train, dataset2stream_processor[dataset_name])

        logger.info('loaded {} samples as debug train/dev/test dataset '.format(debug_examples))

        dev_data = train_data
        test_data = train_data

        if pretrain:
            emb_file = 'glove.6B.50d.txt'
            embeddings = load_embeddings(path.join('data', 'GloVe', emb_file), 'glove')
            logger.info('loaded pre-trained embeddings ({})'.format(emb_file))
            ex.current_run.config["repr_dim_input"] = 50
        else:
            embeddings = Embeddings(None, None)
    else:
        if not use_streaming:
            train_data = load_labelled_data(train)
            dev_data = load_labelled_data(dev)
            test_data = load_labelled_data(test) if test else None
        else:
            s = dataset2stream_processor[dataset_name]
            train_data = load_labelled_data_stream(train, s)
            dev_data = load_labelled_data_stream(dev, s)
            test_data = load_labelled_data_stream(test, s) if test else None

        logger.info('loaded train/dev/test data')
        if pretrain:
            embeddings = load_embeddings(embedding_file, embedding_format)
            logger.info('loaded pre-trained embeddings ({})'.format(embedding_file))
            ex.current_run.config["repr_dim_input"] = embeddings.lookup[0].shape[0]
        else:
            embeddings = Embeddings(None, None)

    vocab = Vocab()
    reader = readers.get_reader_by_name(model)
    parsed_config = ex.current_run.config
    shared_resources = SharedResources(vocab, parsed_config, embeddings)
    reader.configure_with_shared_resources(shared_resources)

    train_reader(reader, train_data, dev_data, test_data, batch_size, clip_value, dataset_name, debug,
                 epochs, l2, learning_rate, learning_rate_decay, log_interval, model, model_dir, tensorboard_folder,
                 use_streaming, validation_interval, write_metrics_to)
