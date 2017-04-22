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

    question_texts = [instance[0].question for instance in train]
    support_texts = [instance[0].support[0] for instance in train]

    candidates_texts = []
    for instance in train:
        candidates_texts += instance[0].atomic_candidates
    answer_texts = [instance[1][0].text for instance in train]

    qs_tokenizer, ca_tokenizer = Tokenizer(), Tokenizer()

    qs_tokenizer.fit_on_texts(question_texts + support_texts)
    ca_tokenizer.fit_on_texts(candidates_texts + answer_texts)

    corpus = {
        'question': qs_tokenizer.texts_to_sequences(question_texts),
        'support': [[s] for s in qs_tokenizer.texts_to_sequences(support_texts)]
    }

    print(corpus)

    sys.exit(0)

    logger.info("Existing models: {}".format(", ".join(readers.readers.keys())))

    config = {
        'batch_size': 128,
        'repr_dim': 128,
        'repr_dim_input': 128,
        'dropout': 0.1
    }

    vocab = Vocab()

    shared_resources = SharedVocabAndConfig(vocab, config)
    reader = JTReader(
        shared_resources,
        SingleSupportFixedClassInputs(shared_resources),
        PairOfBiLSTMOverSupportAndQuestionModel(shared_resources),
        EmptyOutputModule()
    )

    optimizer = tf.train.AdamOptimizer(0.001)
    reader.train(optimizer, train, hooks=[], max_epochs=1, device='/cpu:0')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
