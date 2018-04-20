#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import sys

import tensorflow as tf

from jack.eval import evaluate_reader, pretty_print_results
from jack.io.load import loaders
from jack.readers import reader_from_file

logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_string('dataset', None, 'dataset file')
tf.app.flags.DEFINE_string('loader', 'jack', 'name of loader')
tf.app.flags.DEFINE_string('load_dir', None, 'directory to saved model')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('max_examples', None, 'maximum number of examples to evaluate')
tf.app.flags.DEFINE_string('overwrite', '{}', 'json string that overwrites configuration.')

FLAGS = tf.app.flags.FLAGS

logger.info("Creating and loading reader from {}...".format(FLAGS.load_dir))

kwargs = json.loads(FLAGS.overwrite)

reader = reader_from_file(FLAGS.load_dir, **kwargs)
dataset = loaders[FLAGS.loader](FLAGS.dataset)
if FLAGS.max_examples:
    dataset = dataset[:FLAGS.max_examples]

logger.info("Start!")
result_dict = evaluate_reader(reader, dataset, FLAGS.batch_size)


logger.info("############### RESULTS ##############")
pretty_print_results(result_dict)
