#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import tensorflow as tf

import jtr.jack.readers as readers

from inferte.modules.input import SingleSupportFixedClassInputs
from inferte.modules.model import PairOfBiLSTMOverSupportAndQuestionModel
from inferte.modules.output import EmptyOutputModule
from inferte.reader import JTReader

import tensorflow.contrib.keras as keras

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


def to_corpus(train, qs_tokenizer=None, ca_tokenizer=None):
    question_texts = [instance[0].question for instance in train]
    support_texts = [instance[0].support[0] for instance in train]

    candidates_texts = [instance[0].atomic_candidates for instance in train]
    answer_texts = [instance[1][0].text for instance in train]

    if not qs_tokenizer:
        qs_tokenizer = keras.preprocessing.text.Tokenizer()
        qs_tokenizer.fit_on_texts(question_texts + support_texts)

    if not ca_tokenizer:
        ca_tokenizer = keras.preprocessing.text.Tokenizer()
        ca_tokenizer.fit_on_texts([c for l in candidates_texts for c in l] + answer_texts)

    corpus = {
        'question': qs_tokenizer.texts_to_sequences(question_texts),
        'support': [[s] for s in qs_tokenizer.texts_to_sequences(support_texts)],
        'candidates': [c - 1 for cs in candidates_texts for [c] in ca_tokenizer.texts_to_sequences(cs)],
        'answers': [a - 1 for [a] in ca_tokenizer.texts_to_sequences(answer_texts)]}

    # Note - those parts feel redundant
    corpus['question_lengths'] = [len(q) for q in corpus['question']]
    corpus['support_lengths'] = [[len(s)] for [s] in corpus['support']]
    corpus['ids'] = list(range(len(corpus['question'])))

    return corpus, qs_tokenizer, ca_tokenizer


def main(argv):
    train, dev, test = TestDatasets.generate()
    train = train[:100]

    train_corpus, qs_tokenizer, ca_tokenizer = to_corpus(train)
    dev_corpus, _, _ = to_corpus(dev, qs_tokenizer, ca_tokenizer)
    test_corpus, _, _ = to_corpus(test, qs_tokenizer, ca_tokenizer)

    logger.info("Existing models: {}".format(", ".join(readers.readers.keys())))

    config = {
        'batch_size': 128,
        'repr_dim': 128,
        'repr_dim_input': 128,
        'dropout': 0.1,

        'vocab_size': qs_tokenizer.num_words if qs_tokenizer.num_words else len(qs_tokenizer.word_index) + 1,
        'answer_size': 3
    }

    reader = JTReader(config,
                      SingleSupportFixedClassInputs(),
                      PairOfBiLSTMOverSupportAndQuestionModel(config),
                      EmptyOutputModule())

    optimizer = tf.train.AdamOptimizer(0.001)

    from jtr.jack.train.hooks import LossHook
    hooks = [
        LossHook(reader, iter_interval=10),
        readers.eval_hooks['snli_reader'](reader, dev_corpus, iter_interval=25)
    ]

    reader.train(optimizer, train_corpus, hooks=hooks, max_epochs=500)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
