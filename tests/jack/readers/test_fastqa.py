# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import jack.readers as readers
from jack.core import SharedResources
from jack.io.embeddings.embeddings import Embeddings
from jack.io.load import load_jack
from jack.readers.extractive_qa.util import tokenize
from jack.util.vocab import Vocab


def test_fastqa():
    tf.reset_default_graph()

    data = load_jack('tests/test_data/squad/snippet_jtr.json')
    questions = []
    # fast qa must be initialized with existing embeddings, so we create some
    vocab = dict()
    for question, _ in data:
        questions.append(question)
        for t in tokenize(question.question):
            if t not in vocab:
                vocab[t] = len(vocab)
    embeddings = Embeddings(vocab, np.random.random([len(vocab), 10]))

    # we need a vocabulary (with embeddings for our fastqa_reader, but this is not always necessary)
    vocab = Vocab(vocab=vocab)

    # ... and a config
    config = {"batch_size": 1, "repr_dim": 10, "with_char_embeddings": True}

    # create/setup reader
    shared_resources = SharedResources(vocab, config, embeddings)
    fastqa_reader = readers.fastqa_reader(shared_resources)
    fastqa_reader.setup_from_data(data)

    answers = fastqa_reader(questions)

    assert answers, "FastQA reader should produce answers"
