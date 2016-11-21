import tensorflow as tf
import numpy as np
import random
from batch import augment_with_length, get_feed_dicts
from train import train


def embedder(inputs, input_size, vocab_size, scope=None):
    with tf.variable_scope(scope or "embedder") as varscope:
        embedding_matrix = \
            tf.get_variable("W", [vocab_size, input_size],
                            initializer=tf.random_normal_initializer(0.0, 0.1))
        # [batch_size, max_seq_length, input_size]
        return tf.nn.embedding_lookup(embedding_matrix, inputs)


def reader(inputs, lengths, output_size, contexts=(None, None), scope=None):
    with tf.variable_scope(scope or "reader") as varscope:
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        _, (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            inputs,
            sequence_length=lengths,
            initial_state_fw=contexts[0],
            initial_state_bw=contexts[1],
            dtype=tf.float32
        )

        # each [batch_size x max_seq_length x output_size]
        return states_fw, states_bw


def conditional_reader(seq1, seq1_lengths, seq2, seq2_lengths, output_size, scope=None):
    with tf.variable_scope(scope or "conditional_reader") as varscope:
        # (c_fw, h_fw), (c_bw, h_bw)
        seq1_states = \
            reader(seq1, seq1_lengths, output_size, scope=varscope)
        varscope.reuse_variables()
        # each [batch_size x max_seq_length x output_size]
        return reader(seq2, seq2_lengths, output_size, seq1_states, scope=varscope)


if __name__ == '__main__':
    N = 256
    vocab_size = 10000
    max_len = 20

    input_size = 100
    output_size = 100
    target_size = 3

    batch_size = 64
    num_targets = 3

    np.random.seed(1337)
    random.seed(1337)

    seq1_sampled = []
    for _ in range(0, N):
        len = np.random.randint(2, max_len)
        seq1_sampled.append(np.random.randint(1, vocab_size, [len]))

    seq2_sampled = []
    for _ in range(0, N):
        len = np.random.randint(2, max_len + 2)
        seq2_sampled.append(np.random.randint(1, vocab_size, [len]))

    targets_sampled = []
    for _ in range(0, N):
        targets_sampled.append(np.random.randint(0, num_targets, [1]))

    data = [seq1_sampled, seq2_sampled, targets_sampled]
    data = augment_with_length(data, [0, 1])

    # Model
    # [batch_size, max_seq1_length]
    seq1 = tf.placeholder(tf.int64, [None, None], "seq1")
    # [batch_size]
    seq1_lengths = tf.placeholder(tf.int64, [None], "seq1_lengths")

    # [batch_size, max_seq2_length]
    seq2 = tf.placeholder(tf.int64, [None, None], "seq2")
    # [batch_size]
    seq2_lengths = tf.placeholder(tf.int64, [None], "seq2_lengths")

    # [batch_size]
    targets = tf.placeholder(tf.int64, [None], "targets")

    with tf.variable_scope("embedders") as varscope:
        seq1_embedded = embedder(seq1, input_size, vocab_size)
        varscope.reuse_variables()
        seq2_embedded = embedder(seq2, input_size, vocab_size)

    output = conditional_reader(seq1_embedded, seq1_lengths,
                                seq2_embedded, seq2_lengths,
                                output_size)

    output = tf.concat(1, [output[0][1], output[1][1]])

    logits = tf.nn.rnn_cell._linear(output, target_size, bias=True)
    loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets))
    predict = tf.nn.softmax(logits)

    feed_dicts = \
        get_feed_dicts(data, [seq1, seq1_lengths,
                              seq2, seq2_lengths,
                              targets], batch_size)

    optim = tf.train.AdamOptimizer()

    def report_loss(sess, epoch, iter, predict, loss):
        if iter > 0 and iter % 3 == 0:
            print("epoch %4d\titer %4d\tloss %4.2f" % (epoch, iter, loss))

    hooks = [
        report_loss
    ]

    train(loss, optim, feed_dicts, max_epochs=1000, hooks=hooks)



