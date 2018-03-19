#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import sys

import tensorflow as tf

from jack.io.load import loaders
from jack.readers.implementations import reader_from_file

logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_string('dataset', None, 'dataset file')
tf.app.flags.DEFINE_string('loader', 'squad', 'either squad or jack')
tf.app.flags.DEFINE_string('load_dir', None, 'directory to saved model')
tf.app.flags.DEFINE_string('out', "results.json", 'Result file path.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_string('overwrite', '{}', 'json string that can overwrite configuration.')

FLAGS = tf.app.flags.FLAGS

logger.info("Creating and loading reader from {}...".format(FLAGS.load_dir))
config = {"max_support_length": None}
config.update(json.loads(FLAGS.overwrite))
reader = reader_from_file(FLAGS.load_dir, **config)

dataset = loaders[FLAGS.loader](FLAGS.file)

logger.info("Start!")
answers = reader.process_dataset(dataset, FLAGS.batch_size, silent=False)
results = {dataset[i][0].id: a.text for i, a in enumerate(answers)}
with open(FLAGS.out, "w") as out_file:
    json.dump(results, out_file)

logger.info("Done!")
