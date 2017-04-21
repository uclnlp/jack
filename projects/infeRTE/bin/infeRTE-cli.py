#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf

import jtr.jack.readers as readers

from inferte.modules.input import SingleSupportFixedClassInputs
from inferte.modules.model import PairOfBiLSTMOverSupportAndQuestionModel
from inferte.modules.output import EmptyOutputModule

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestDatasets(object):
    @staticmethod
    def generate():
        from jtr.jack.data_structures import load_labelled_data
        splits = ['train.json', 'dev.json', 'test.json']
        return [load_labelled_data(os.path.join('SNLI/', split)) for split in splits]

logger.info("Existing models: {}".format(", ".join(readers.readers.keys())))

config = {
    'batch_size': 128,
    'repr_dim': 128,
    'repr_dim_input': 128,
    'dropout': 0.1
}

vocab = Vocab()

shared_resources = SharedVocabAndConfig(vocab, config)
reader = JTReader(shared_resources,
                  SingleSupportFixedClassInputs(shared_resources),
                  PairOfBiLSTMOverSupportAndQuestionModel(shared_resources),
                  EmptyOutputModule())

print(vocab.sym2id)

train, dev, test = TestDatasets.generate()
train = train[:10]

print(len(train))
print(train[0])

# We creates hooks which keep track of the loss
# We also create 'the standard hook' for our model
from jtr.jack.train.hooks import LossHook
hooks = [
    LossHook(reader, iter_interval=10),
    readers.eval_hooks['snli_reader'](reader, dev, iter_interval=25)
]

# Here we initialize our optimizer
# we choose Adam with standard momentum values and learning rate 0.001
learning_rate = 0.001
optim = tf.train.AdamOptimizer(learning_rate)

# Lets train the reader on the CPU for 2 epochs
reader.train(optim, train,
             hooks=hooks,
             max_epochs=1,
             device='/cpu:0')

print(vocab.sym2id)
