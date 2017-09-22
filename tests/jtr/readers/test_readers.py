"""Smoke test: train all readers for one iteration & run inference."""
from functools import partial

from jtr import readers
from jtr.core import SharedResources, JTReader
from jtr.data_structures import QASetting, Answer
from jtr.input_output.embeddings import Vocabulary, Embeddings
from jtr.tasks.xqa.util import tokenize
from jtr.util.vocab import Vocab

import tensorflow as tf
import numpy as np


def teardown_function(_):
    tf.reset_default_graph()


def build_vocab(questions):
    """Since some readers require an initilized vocabulary, initialize it here."""

    vocab = dict()
    for question in questions:
        for t in tokenize(question.question):
            if t not in vocab:
                vocab[t] = len(vocab)
    vocab = Vocabulary(vocab)
    embeddings = Embeddings(vocab, np.random.random([len(vocab), 10]))

    vocab = Vocab.create_from_embeddings(embeddings)
    return vocab, embeddings


def smoke_test(reader_name):
    """Instantiate the reader, train for one epoch, and run inference."""

    data_set = [
        (QASetting(
            question="Which is it?",
            support=["While b seems plausible, answer a is correct."],
            id="1",
            atomic_candidates=["a", "b", "c"]),
         [Answer("a", (0, 6, 6))])
    ]
    questions = [q for q, _ in data_set]

    vocab, embeddings = build_vocab(questions)
    shared_resources = SharedResources(vocab, {"repr_dim": 10,
                                               "repr_dim_input": 10,
                                               "dropout": 0.5,
                                               "batch_size": 1}, embeddings=embeddings)

    reader = readers.readers[reader_name](shared_resources)

    reader.train(tf.train.AdamOptimizer(), data_set, max_epochs=1)

    answers = reader(questions)

    assert answers, "%s should produce answers" % reader_name


# TODO: Make streaming work as well.
BLACKLIST = ["cbilstm_snli_streaming_reader"]
READERS = [r for r in readers.readers.keys()
           if r not in BLACKLIST]

# Dynamically generate one test for each reader
current_module = __import__(__name__)

for reader_name in READERS:
    setattr(current_module, "test_%s" % reader_name,
            partial(smoke_test, reader_name))
