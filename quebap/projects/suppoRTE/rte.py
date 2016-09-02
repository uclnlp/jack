import tensorflow as tf
import numpy as np


def reader(input, length, input_size, hidden_size, vocab_size,
           cell_constructor=tf.nn.rnn_cell.LSTMCell):
    """
    :param input: [batch_size x seq_length]
    :param length:
    :param input_size:
    :param hidden_size:
    :param vocab_size:
    :param cell_constructor:
    :return:
    """

    embedding_matrix = tf.Variable(
        tf.random_uniform([vocab_size, input_size], -0.1, 0.1),
        trainable=True
    )

    # [batch_size, max_seq_length, input_size]
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, input)

    cell_fw = cell_constructor(input_size, hidden_size, state_is_tuple=True)
    cell_bw = cell_constructor(input_size, hidden_size, state_is_tuple=True)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        embedded_inputs,
        sequence_length=length,
        dtype=tf.float32
    )

    return outputs


if __name__ == '__main__':
    rand = np.random

    batch_size = 3
    seq_length = 7
    vocab_size = 5

    hidden_size = 2

    premise = rand.randint(1, vocab_size, (batch_size, seq_length))
    premise_length = rand.randint(3, seq_length, batch_size)
    hypothesis = rand.randint(1, vocab_size, (batch_size, seq_length))
    hypothesis_length = rand.randint(3, seq_length, batch_size)
    target = rand.randint(0, 3, batch_size)

    premise_placeholder = tf.placeholder(tf.int64, shape=(None, None), name="premise")
    premise_length_placeholder = tf.placeholder(tf.int64, shape=None, name="premise_length")
    hypothesis_placeholder = tf.placeholder(tf.int64, shape=(None, None), name="hypothesis")
    hypothesis_length_placeholder = tf.placeholder(tf.int64, shape=None, name="hypothesis_length")
    target_placeholder = tf.placeholder(tf.int64, shape=None, name="target")

    outputs = reader(premise_placeholder, premise_length_placeholder,
                     hidden_size, hidden_size, vocab_size)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(outputs, {
            premise_placeholder: premise,
            premise_length_placeholder: premise_length
        }))


