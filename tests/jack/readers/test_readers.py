# -*- coding: utf-8 -*-

"""Smoke test: train all readers for one iteration & run inference."""

from functools import partial

import numpy as np
import tensorflow as tf

from jack import readers
from jack.core.data_structures import QASetting, Answer
from jack.core.shared_resources import SharedResources
from jack.core.tensorflow import TFReader
from jack.io.embeddings import Embeddings
from jack.readers.extractive_qa.util import tokenize
from jack.util.vocab import Vocab


def teardown_function(_):
    tf.reset_default_graph()


def build_vocab(questions):
    """Since some readers require an initialized vocabulary, initialize it here."""

    vocab = dict()
    for question in questions:
        for t in tokenize(question.question):
            if t not in vocab:
                vocab[t] = len(vocab)
    embeddings = Embeddings(vocab, np.random.random([len(vocab), 10]))

    vocab = Vocab(vocab=embeddings.vocabulary)
    return vocab, embeddings


def smoke_test(reader_name):
    """Instantiate the reader, train for one epoch, and run inference."""

    data_set = [
        (QASetting(
            question="Which is it?",
            support=["While b seems plausible, answer a is correct."],
            id="1",
            candidates=["a", "b", "c"]),
         [Answer("a", (6, 6))])
    ]
    questions = [q for q, _ in data_set]
    v, e = build_vocab(questions)
    shared_resources = SharedResources(v, {"repr_dim": 10, "dropout": 0.5}, e)
    tf.reset_default_graph()
    reader = readers.readers[reader_name](shared_resources)
    if isinstance(reader, TFReader):
        reader.train(tf.train.AdamOptimizer(), data_set, batch_size=1, max_epochs=1)
    else:
        import torch
        reader.setup_from_data(data_set, is_training=True)
        params = list(reader.model_module.prediction_module.parameters())
        params.extend(reader.model_module.loss_module.parameters())
        optimizer = torch.optim.Adam(params, lr=0.01)
        reader.train(optimizer, data_set, batch_size=1, max_epochs=1)

    answers = reader(questions)

    assert answers, "{} should produce answers".format(reader_name)


BLACKLIST = ['fastqa_reader_torch', 'modular_qa_reader', 'modular_nli_reader']
READERS = [r for r in readers.readers.keys()
           if r not in BLACKLIST]

# Dynamically generate one test for each reader
current_module = __import__(__name__)

for reader_name in READERS:
    setattr(current_module, "test_{}".format(reader_name), partial(smoke_test, reader_name))
