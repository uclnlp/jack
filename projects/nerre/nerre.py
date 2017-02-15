import os
from collections import defaultdict
from typing import NamedTuple, Sequence, Mapping

import tensorflow as tf

# tensorflow inputs
from jtr.preprocess.vocab import Vocab
from projects.nerre.data import read_ann

tokens = tf.placeholder(tf.int32, [None, None])  # [batch_size, max_num_tokens]
sentence_lengths = tf.placeholder(tf.int32, [None])
target_start_end_labels = tf.placeholder(tf.int32, [None, None])  # [batch_size, max_num_tokens]
target_relations = tf.placeholder(tf.int32, [None, None, None])  # [batch_size, max_num_tokens, max_num_tokens]]

START = 0
END = 1
OUTSIDE = 2

HYPONYM_OF = 0
SYNONYM_OF = 1


def create_model(output_size, layers, dropout, num_words, emb_dim):
    with tf.variable_scope("embeddings"):
        embeddings = tf.get_variable("embeddings", shape=[num_words, emb_dim], dtype=tf.float32)

    with tf.variable_scope("input"):
        embedded_input = tf.gather(embeddings, tokens)

    with tf.variable_scope("model"):
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        if layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers)

        if dropout != 0.0:
            cell_dropout = \
                tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0 - dropout)
        else:
            cell_dropout = cell

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_dropout,
            cell_dropout,
            embedded_input,
            sequence_length=sentence_lengths,
            dtype=tf.float32
        )



        # fw = states[0][1]
        #
        # # todo: also use backward pass
        # # bw = states[1][1]
        #
        # h = fw
        #
        # logits = tf.contrib.layers.linear(h, target_size)
        #
        # predict = tf.arg_max(tf.nn.softmax(logits), 1)
        #
        # loss = tf.reduce_sum(
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order))
        #
        # return loss, placeholders, predict

train_dir = "/Users/riedel/corpora/scienceie/train2"
dev_dir = "/Users/riedel/corpora/scienceie/dev/"

train_instances = read_ann(train_dir)
dev_instances = read_ann(dev_dir)

print("Loaded {} training instances".format(len(train_instances)))

vocab = Vocab()

for instance in train_instances + dev_instances:
    for sent in instance.doc:
        for token in sent.tokens:
            vocab(token.word)

print("Collected {} word types".format(len(vocab)))