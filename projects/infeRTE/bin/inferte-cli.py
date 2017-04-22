#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf

import jtr.jack.readers as readers

#from jtr.jack.tasks.mcqa.simple_mcqa import SingleSupportFixedClassInputs
#from jtr.jack.tasks.mcqa.simple_mcqa import PairOfBiLSTMOverSupportAndQuestionModel
#from jtr.jack.tasks.mcqa.simple_mcqa import EmptyOutputModule
#from jtr.jack.core import JTReader

from inferte.modules.input import SingleSupportFixedClassInputs
from inferte.modules.model import PairOfBiLSTMOverSupportAndQuestionModel
from inferte.modules.output import EmptyOutputModule
from inferte.reader import JTReader

from inferte.preprocessing.text import Tokenizer

from jtr.preprocess.vocab import Vocab
from jtr.jack.core import SharedVocabAndConfig

import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestDatasets:
    @staticmethod
    def generate():
        from jtr.jack.data_structures import load_labelled_data
        splits = ['train.json', 'dev.json', 'test.json']
        return [load_labelled_data(os.path.join('SNLI/', split)) for split in splits]


def main(argv):
    train, dev, test = TestDatasets.generate()
    train = train[:10]

    print(train)

    sys.exit(0)

    texts = ['Hello world', 'How are you?']
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)

    print('seqs', seqs)

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

    print(train[0])

    optimizer = tf.train.AdamOptimizer(0.001)
    reader.train(optimizer, train, hooks=[], max_epochs=1, device='/cpu:0')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
