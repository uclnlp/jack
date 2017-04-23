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

from inferte.preprocessing.text import Tokenizer

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

    candidates_texts = [instance[0].atomic_candidates for instance in train]
    answer_texts = [instance[1][0].text for instance in train]

    qs_tokenizer, ca_tokenizer = Tokenizer(), Tokenizer()

    qs_tokenizer.fit_on_texts(question_texts + support_texts)
    ca_tokenizer.fit_on_texts([c for l in candidates_texts for c in l] + answer_texts)

    corpus = {
        'question': qs_tokenizer.texts_to_sequences(question_texts),
        'support': [[s] for s in qs_tokenizer.texts_to_sequences(support_texts)],
        'candidates': [c for cs in candidates_texts for [c] in ca_tokenizer.texts_to_sequences(cs)],
        'answers': [a for [a] in ca_tokenizer.texts_to_sequences(answer_texts)]}

    # Note - those parts feel redundant
    corpus['question_lengths'] = [len(q) for q in corpus['question']]
    corpus['support_lengths'] = [[len(s)] for [s] in corpus['support']]
    corpus['ids'] = list(range(len(corpus['question'])))

    logger.info("Existing models: {}".format(", ".join(readers.readers.keys())))

    config = {
        'batch_size': 128,
        'repr_dim': 128,
        'repr_dim_input': 128,
        'dropout': 0.1,

        'vocab_size': qs_tokenizer.num_words if qs_tokenizer.num_words else len(qs_tokenizer.word_index) + 1,
        'answer_size': 4
    }

    reader = JTReader(config,
                      SingleSupportFixedClassInputs(),
                      PairOfBiLSTMOverSupportAndQuestionModel(config),
                      EmptyOutputModule())

    optimizer = tf.train.AdamOptimizer(0.001)
    reader.train(optimizer, corpus, hooks=[], max_epochs=1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
