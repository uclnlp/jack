import tensorflow as tf


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
    with tf.variable_scope(scope or "conditional_reader_seq1") as varscope1:
        # (c_fw, h_fw), (c_bw, h_bw)
        seq1_states = \
            reader(seq1, seq1_lengths, output_size, scope=varscope1)
    #with tf.variable_scope(scope or "conditional_reader_seq2") as varscope2:
        varscope1.reuse_variables()
        # each [batch_size x max_seq_length x output_size]
        return reader(seq2, seq2_lengths, output_size, seq1_states, scope=varscope1)


def predictor(output, targets, target_size):
    logits = tf.contrib.layers.fully_connected(output, target_size)
    loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets))
    predict = tf.arg_max(tf.nn.softmax(logits), 1)
    return logits, loss, predict


def conditional_reader_model(input_size, output_size, vocab_size, target_size):
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

    logits, loss, predict = predictor(output, targets, target_size)

    return (logits, loss, predict), \
           (seq1, seq1_lengths, seq2, seq2_lengths, targets)  # placeholders
