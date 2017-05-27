# -*- coding: utf-8 -*-

from sacred import Experiment
from sacred.arg_parser import parse_args
import sys

parsed_args = dict([x.split("=") for x in parse_args(sys.argv)["UPDATE"]])
if "config" in parsed_args:
    path = parsed_args["config"]
else:
    path = "./conf/jack.yaml"


def fetch_parents(path, parents=[]):
    tmp_ex = Experiment('jack')
    tmp_ex.add_config(path)
    tmp_ex.run("print_config")
    if tmp_ex.current_run is not None and "parent_config" in tmp_ex.current_run.config:
        return fetch_parents(tmp_ex.current_run.config["parent_config"], [path] + parents)
    else:
        return [path] + parents

configs = fetch_parents(path)
print("Loading", configs)
ex = Experiment('jack')
for path in configs:
    ex.add_config(path)

import os
import os.path as path
import logging
import math

import random
import shutil
from time import time

import tensorflow as tf
from tensorflow.python.client import device_lib

import jtr.jack.readers as readers
from jtr.jack.data_structures import load_labelled_data
from jtr.jack.train.hooks import LossHook, ExamplesPerSecHook, ETAHook
from jtr.load.embeddings.embeddings import load_embeddings, Embeddings
from jtr.preprocess.vocab import Vocab
from jtr.jack.core import SharedVocabAndConfig

logger = logging.getLogger(os.path.basename(sys.argv[0]))


class Duration(object):
    def __init__(self):
        self.t0 = time()
        self.t = time()

    def __call__(self):
        logger.info('Time since last checkpoint : {0:.2g}min'.format((time() - self.t) / 60.))
        self.t = time()


checkpoint = Duration()

logging.basicConfig(level=logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # print only TF errors


@ex.automain
def main(batch_size,
         clip_value,
         config,
         debug,
         debug_examples,
         dev,
         embedding_file,
         embedding_format,
         epochs,
         l2,
         learning_rate,
         learning_rate_decay,
         log_interval,
         model,
         model_dir,
         pretrain,
         seed,
         tensorboard_folder,
         test,
         train,
         vocab_from_embeddings,
         write_metrics_to):

    # make everything deterministic
    random.seed(seed)
    tf.set_random_seed(seed)

    print(config)

    if clip_value != 0.0:
        clip_value = - abs(clip_value), abs(clip_value)

    # Get information about available CPUs and GPUs:
    # to set specific device, add CUDA_VISIBLE_DEVICES environment variable, e.g.
    # $ CUDA_VISIBLE_DEVICES=0 ./jtr_script.py

    logger.info('available devices:')
    for device in device_lib.list_local_devices():
        logger.info('device info: ' + str(device).replace("\n", " "))

    if debug:
        train_data = load_labelled_data(train, debug_examples)

        logger.info('loaded {} samples as debug train/dev/test dataset '.format(debug_examples))

        dev_data = train_data
        test_data = train_data
        if pretrain:
            emb_file = 'glove.6B.50d.txt'
            embeddings = load_embeddings(path.join('data', 'GloVe', emb_file), 'glove')
            logger.info('loaded pre-trained embeddings ({})'.format(emb_file))
        else:
            embeddings = Embeddings(None, None)
    else:
        train_data, dev_data = [load_labelled_data(name) for name in [train, dev]]
        test_data = load_labelled_data(test) if test else None
        logger.info('loaded train/dev/test data')
        if pretrain:
            embeddings = load_embeddings(embedding_file, embedding_format)
            logger.info('loaded pre-trained embeddings ({})'.format(embedding_file))
        else:
            embeddings = Embeddings(None, None)

    emb = embeddings

    vocab = Vocab(emb=emb, init_from_embeddings=vocab_from_embeddings)

    # build JTReader
    checkpoint()

    parsed_config = ex.current_run.config

    shared_resources = SharedVocabAndConfig(vocab, parsed_config, train_data)
    reader = readers.readers[model](shared_resources)
    checkpoint()

    learning_rate = tf.get_variable("learning_rate", initializer=learning_rate, dtype=tf.float32,
                                    trainable=False)
    lr_decay_op = learning_rate.assign(learning_rate_decay * learning_rate)
    optim = tf.train.AdamOptimizer(learning_rate)

    if tensorboard_folder is not None:
        if os.path.exists(tensorboard_folder):
            shutil.rmtree(tensorboard_folder)
        sw = tf.summary.FileWriter(tensorboard_folder)
    else:
        sw = None

    # Hooks
    iter_interval = 1 if debug else log_interval
    hooks = [LossHook(reader, iter_interval, summary_writer=sw),
             ExamplesPerSecHook(reader, batch_size, iter_interval, sw),
             ETAHook(reader, iter_interval, math.ceil(len(train_data) / batch_size), epochs,
                     checkpoint(), sw)]

    preferred_metric, best_metric = readers.eval_hooks[model].preferred_metric_and_best_score()

    def side_effect(metrics, prev_metric):
        """Returns: a state (in this case a metric) that is used as input for the next call"""
        m = metrics[preferred_metric]
        if prev_metric is not None and m < prev_metric:
            reader.sess.run(lr_decay_op)
            logger.info("Decayed learning rate to: %.5f" % reader.sess.run(learning_rate))
        elif m > best_metric[0]:
            best_metric[0] = m
            if prev_metric is None:  # store whole model only at beginning of training
                reader.store(model_dir)
            else:
                reader.model_module.store(reader.sess, os.path.join(model_dir, "model_module"))
            logger.info("Saving model to: %s" % model_dir)
        return m

    # this is the standard hook for the model
    hooks.append(readers.eval_hooks[model](
        reader, dev_data, summary_writer=sw, side_effect=side_effect,
        iter_interval=checkpoint(),
        epoch_interval=(1 if checkpoint is None else None),
        write_metrics_to=write_metrics_to))

    # Train
    reader.train(optim, training_set=train_data,
                 max_epochs=epochs, hooks=hooks,
                 l2=l2, clip=clip_value, clip_op=tf.clip_by_value)

    # Test final model
    if test_data is not None:
        test_eval_hook = readers.eval_hooks[model](
            reader, test_data, summary_writer=sw, epoch_interval=1, write_metrics_to=write_metrics_to)

        reader.load(model_dir)
        test_eval_hook.at_test_time(1)