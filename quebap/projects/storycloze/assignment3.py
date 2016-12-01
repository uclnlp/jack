import json
from pprint import pprint
from time import time, sleep
from os import path

from math import factorial
from quebap.sisyphos.batch import get_feed_dicts
from quebap.sisyphos.vocab import Vocab, NeuralVocab
from quebap.sisyphos.map import tokenize, lower, deep_map, deep_seq_map
from quebap.sisyphos.models import conditional_reader_model
from quebap.sisyphos.train import train
from quebap.sisyphos.prepare_embeddings import load as loads_embeddings
import tensorflow as tf
import numpy as np
import random
from quebap.sisyphos.hooks import SpeedHook, AccuracyHook, LossHook


def load_corpus(name):
    story = []
    order = []

    with open("./quebap/data/StoryCloze/%s_shuffled.tsv" % name, "r") as f:
        for line in f.readlines():
            splits = [x.strip() for x in line.split("\t")]
            current_story = splits[0:5]
            current_order = splits[5:]

            story.append(current_story)
            order.append(permutation_index(current_order))

    return {"story": story, "order": order}


def permutation_index(p):
    result = 0
    for j in range(len(p)):
        k = sum(1 for i in p[j + 1:] if i < p[j])
        result += k * factorial(len(p) - j - 1)
    return result


def pipeline(corpus, vocab=None, target_vocab=None, emb=None, freeze=False,
             concat_seq=True):
    vocab = vocab or Vocab(emb=emb)
    target_vocab = target_vocab or Vocab(unk=None)
    if freeze:
        vocab.freeze()
        target_vocab.freeze()

    corpus_tokenized = deep_map(corpus, tokenize, ["story"])
    corpus_lower = deep_seq_map(corpus_tokenized, lower, ["story"])
    corpus_os = deep_seq_map(corpus_lower,
                             lambda xs: ["<SOS>"] + xs + ["<EOS>"], ["story"])
    corpus_ids = deep_map(corpus_os, vocab, ["story"])
    corpus_ids = deep_map(corpus_ids, target_vocab, ["order"])
    if concat_seq:
        for i in range(len(corpus_ids["story"])):
            corpus_ids["story"][i] = [x for xs in corpus_ids["story"][i] for x in xs]
    corpus_ids = deep_seq_map(corpus_ids,
                              lambda xs: len(xs), keys=["story"],
                              fun_name='length', expand=True)
    return corpus_ids, vocab, target_vocab


def get_model(vocab_size, input_size, output_size):
    # Model

    # Placeholders
    # [batch_size x max_length]
    story = tf.placeholder(tf.int64, [None, None], "story")
    # [batch_size]
    story_length = tf.placeholder(tf.int64, [None], "story_length")
    # [batch_size]
    order = tf.placeholder(tf.int64, [None], "order")

    placeholders = {"story": story, "story_length": story_length, "order": order}

    # Word embeddings
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    embeddings = tf.get_variable("W", [vocab_size, input_size],
                                 initializer=initializer)
    # [batch_size x max_seq_length x input_size]
    story_embedded = tf.nn.embedding_lookup(embeddings, story)

    with tf.variable_scope("reader") as varscope:
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            story_embedded,
            sequence_length=story_length,
            dtype=tf.float32
        )

        final_state = states[-1]

        return final_state, placeholders


if __name__ == '__main__':
    # Config
    DEBUG = True
    INPUT_SIZE = 100
    OUTPUT_SIZE = 100
    BATCH_SIZE = 8

    LEARNING_RATE = 0.01

    if DEBUG:
        train_corpus = load_corpus("debug")
        dev_corpus = load_corpus("debug")
        test_corpus = load_corpus("debug")
    else:
        train_corpus = load_corpus("train")
        dev_corpus = load_corpus("dev")
        test_corpus = load_corpus("test")

    train_mapped, train_vocab, train_target_vocab = pipeline(train_corpus)
    dev_mapped, _, _ = pipeline(dev_corpus, train_vocab, train_target_vocab)
    test_mapped, _, _ = pipeline(test_corpus, train_vocab, train_target_vocab)

    final_state, placeholders = get_model(len(train_vocab), INPUT_SIZE,
                                             OUTPUT_SIZE)

    # Training
    train_feed_dicts = get_feed_dicts(train_mapped, placeholders, BATCH_SIZE)
    dev_feed_dicts = get_feed_dicts(dev_mapped, placeholders, BATCH_SIZE)
    test_feed_dicts = get_feed_dicts(test_mapped, placeholders, BATCH_SIZE)

    optim = tf.train.AdamOptimizer(LEARNING_RATE)

    hooks = [
        LossHook(100, BATCH_SIZE),
        SpeedHook(100, BATCH_SIZE),
        #AccuracyHook(dev_feed_dicts, predict, placeholders['targets'], 2)
    ]

    #train(loss, optim, train_feed_dicts, max_epochs=1000, hooks=hooks)

    for batch in train_feed_dicts:
        print(batch)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for batch in train_feed_dicts:

            result = sess.run(final_state, batch)
            print(result)

