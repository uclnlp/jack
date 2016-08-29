import tensorflow as tf
import numpy as np
import quebap.util.tfutil as tfutil


def dropout_noiserizer(keep_prob=0.5):
    def fun(inputs):
        """
        :param inputs:
        :return:
        """
        return tf.nn.dropout(inputs, keep_prob)

    return fun


def embedder(inputs, input_size, noiserizer=dropout_noiserizer):
    """
    :param inputs:
    :param input_size:
    :return:
    """
    with tf.variable_scope("embedder",
                           initializer=tf.random_normal_initializer()):
        embedding_matrix = \
            tf.get_variable("embedding_matrix", shape=(vocab_size, input_size),
                            trainable=True)

    # [batch_size x max_seq_length x input_size]
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)

    return noiserizer(embedded_inputs)


def text2vecs(embedded_inputs, seq_lengths, hidden_size,
              cell_constructor=tf.nn.rnn_cell.LSTMCell,
              bidirectional=True, projected_fwbw=True):
    """
    :param embedded_inputs: [batch_size x max_seq_length x input_size]
    :param seq_lengths: [batch_size]
    :param input_size:
    :param hidden_size:
    :param cell_constructor:
    :param bidirectional:
    :return:
      outputs [batch_size x max_seq_length x output_dim]
    """

    cell = cell_constructor(num_units=hidden_size, state_is_tuple=True)

    if bidirectional:
        outputs_fwbw_tuple, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            dtype=tf.float32,
            sequence_length=seq_lengths,
            inputs=embedded_inputs)
        outputs_fwbw = tf.concat(2, outputs_fwbw_tuple)
        if projected_fwbw:
            outputs_fwbw_flattened = tf.reshape(outputs_fwbw,
                                                shape=(-1, 2*hidden_size))
            outputs_flattened = \
                tf.contrib.layers.fully_connected(outputs_fwbw_flattened,
                                                  hidden_size,
                                                  activation_fn=tf.nn.tanh),
            outputs = \
                tf.add_n([outputs_fwbw_tuple[0], outputs_fwbw_tuple[1],
                          tf.reshape(outputs_flattened,
                                     shape=tf.shape(outputs_fwbw_tuple[0]))])
        else:
            outputs = outputs_fwbw
    else:
        outputs, _ = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=seq_lengths,
            inputs=embedded_inputs)

    return outputs


def symbolizer(outputs, vocab_size):
    """
    :param outputs: [batch_size x max_seq_length x output_dim]
    :return:
    """
    return tf.contrib.layers.fully_connected(outputs, vocab_size,
                                             activation_fn=None)


def unsupervised_loss(logits, targets, seq_lengths):
    """
    :param logits: [batch_size x max_seq_length x vocab_size]
    :param target: [batch_size x max_seq_length]
    :param seq_lengths: [batch_size]
    :return:
    """
    mask = tfutil.mask_for_lengths(seq_lengths, mask_right=False, value=1.0)
    mask_reshaped = tf.reshape(mask, shape=(-1,))
    logits_reshaped = tf.reshape(logits, shape=(-1, vocab_size))
    targets_reshaped = tf.reshape(targets, shape=(-1,))

    # return tf.nn.softmax_cross_entropy_with_logits(masked_logits, targets)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits_reshaped, targets_reshaped
    )
    loss_masked = loss * mask_reshaped
    return tf.reduce_sum(loss_masked) / tf.reduce_sum(mask_reshaped)


if __name__ == '__main__':
    input_size = 7
    hidden_size = 7
    vocab_size = 20
    batch_size = 3
    max_seq_length = 5

    # [mb x seq_length]
    inputs = np.random.randint(0, vocab_size, size=(batch_size, max_seq_length))
    seq_lengths = np.random.randint(2, max_seq_length + 1, batch_size)

    inputs_sliced = tf.slice(inputs, (0, 0), tf.pack(
        [-1, tf.cast(tf.reduce_max(seq_lengths), tf.int32)]
    ))

    with tf.variable_scope("autoread",
                           initializer=tf.contrib.layers.xavier_initializer()):

        keep_prob = tf.get_variable("keep_prob", shape=[], trainable=False,
                                    initializer=tf.constant_initializer(0.5))

        embedded_inputs = embedder(inputs_sliced, input_size,
                                   dropout_noiserizer(keep_prob))

        outputs = text2vecs(embedded_inputs, seq_lengths, hidden_size)
        logits = symbolizer(outputs, vocab_size)
        loss = unsupervised_loss(logits, inputs_sliced, seq_lengths)
        symbols = tf.argmax(logits, 2)

        optim_op = tf.train.AdamOptimizer().minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for _ in range(1000):
                _, loss_current, symbols_current = \
                    sess.run([optim_op, loss, symbols])
                print("inputs:\n%s\n\nsymbols:\n%s\n\nloss: %.3f\n" %
                      (str(inputs), str(symbols_current), loss_current))
