#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import sys

import tensorflow as tf

from jack.io.load import loaders
from jack.readers import reader_from_file, eval_hooks

logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_string('dataset', None, 'dataset file')
tf.app.flags.DEFINE_string('loader', 'jack', 'name of loader')
tf.app.flags.DEFINE_string('model_dir', None, 'directory to saved model')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_string('overwrite', '', 'json string that overwrites configuration.')

FLAGS = tf.app.flags.FLAGS

logger.info("Creating and loading reader from {}...".format(FLAGS.model_dir))

kwargs = json.loads(FLAGS.overwrite)

reader = reader_from_file(FLAGS.model_dir, **kwargs)
dataset = loaders[FLAGS.loader](FLAGS.dataset)

logger.info("Start!")

def side_effect(metrics, _):
    """Returns: a state (in this case a metric) that is used as input for the next call"""
    logger.info("#####################################")
    logger.info("Results:")
    for k, v in metrics.items():
        logger.info("{}: {}".format(k, v))
    logger.info("#####################################")
    return 0.0

test_eval_hook = eval_hooks[reader.shared_resources.config["model"]](
    reader, dataset, FLAGS.batch_size, epoch_interval=1, side_effect=side_effect)
test_eval_hook.at_test_time(1)

logger.info("Done!")
