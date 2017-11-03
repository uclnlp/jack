#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import json

import tensorflow as tf

from jack.data_structures import jtr_to_qasetting
from jack.io.SQuAD2jtr import convert_squad
from jack.io.embeddings import load_embeddings
from jack.readers import readers
from jack.util.vocab import Vocab

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_string('file', None, 'dataset file')
tf.app.flags.DEFINE_string('dataset_type', 'squad', 'either squad or jack')
tf.app.flags.DEFINE_string('model', None, 'Name of the reader')
tf.app.flags.DEFINE_string('model_dir', None, 'directory to saved model')
tf.app.flags.DEFINE_string('embedding_path', None, 'path to embeddings')
tf.app.flags.DEFINE_string('embedding_format', 'glove', 'embeddings format')
tf.app.flags.DEFINE_string('device', "/cpu:0", 'device to use')
tf.app.flags.DEFINE_string('out', "results.json", 'Result file path.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('beam_size', 1, 'beam size')
tf.app.flags.DEFINE_string('kwargs', '{}', 'additional reader-specific configurations')

FLAGS = tf.app.flags.FLAGS

logger.info("Loading embeddings from {}...".format(FLAGS.embedding_path))
emb = load_embeddings(FLAGS.embedding_path, FLAGS.embedding_format)
vocab = Vocab(emb=emb, init_from_embeddings=True)

logger.info("Creating and loading reader from {}...".format(FLAGS.model_dir))
config = {"beam_size": FLAGS.beam_size, 'batch_size': FLAGS.batch_size, "max_support_length": None}
config.update(json.loads(FLAGS.kwargs))
reader = readers[FLAGS.model](vocab, config)
with tf.device(FLAGS.device):
    reader.load_and_setup(FLAGS.model_dir)

if FLAGS.dataset_type == "squad":
    dataset_jtr = convert_squad(FLAGS.file)
else:
    with open(FLAGS.file) as f:
        dataset_jtr = json.load(f)

dataset = jtr_to_qasetting(dataset_jtr)

logger.info("Start!")
answers = reader.process_dataset(dataset, FLAGS.batch_size, debug=True)
results = {dataset[i][0].id: a.text for i, a in enumerate(answers)}
with open(FLAGS.out, "w") as out_file:
    json.dump(results, out_file)

logger.info("Done!")
