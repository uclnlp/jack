import tensorflow as tf


def get_permute_model(vocab_size, input_size, output_size, target_size, layers=1,
                      dropout=0.0):
    # Placeholders
    # [batch_size x max_length]
    story = tf.placeholder(tf.int64, [None, None], "story")
    # [batch_size]
    story_length = tf.placeholder(tf.int64, [None], "story_length")
    # [batch_size]
    order = tf.placeholder(tf.int64, [None], "order")
    placeholders = {"story": story, "story_length": story_length,
                    "order": order}

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

        if layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers)

        if dropout != 0.0:
            cell_dropout = \
                tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0-dropout)
        else:
            cell_dropout = cell

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_dropout,
            cell_dropout,
            story_embedded,
            sequence_length=story_length,
            dtype=tf.float32
        )

        c, h = states[-1]  # LSTM state is a tuple

        logits = tf.contrib.layers.linear(h, target_size)

        predict = tf.arg_max(tf.nn.softmax(logits), 1)

        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order))

        return loss, placeholders, predict


def get_basic_model(vocab_size, input_size, output_size, target_size, layers=1,
                    dropout=0.0, nvocab=None):
    # Placeholders
    # [batch_size x max_length]
    story = tf.placeholder(tf.int64, [None, None], "story")
    # [batch_size]
    story_length = tf.placeholder(tf.int64, [None], "story_length")
    # [batch_size x 5]
    order = tf.placeholder(tf.int64, [None, None], "order")
    placeholders = {"story": story, "story_length": story_length,
                    "order": order}

    # Word embeddings

    if nvocab is None:
        initializer = tf.random_uniform_initializer(-0.05, 0.05)
        embeddings = tf.get_variable("W", [vocab_size, input_size],
                                     initializer=initializer)
    else:
        print('..using pretrained embeddings')
        embeddings = nvocab.embedding_matrix

    # [batch_size x max_seq_length x input_size]
    story_embedded = tf.nn.embedding_lookup(embeddings, story)

    with tf.variable_scope("reader") as varscope:
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        if layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers)

        if dropout != 0.0:
            cell_dropout = \
                tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0-dropout)
        else:
            cell_dropout = cell

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_dropout,
            cell_dropout,
            story_embedded,
            sequence_length=story_length,
            dtype=tf.float32
        )

        c, h = states[-1]  # LSTM state is a tuple

        # [batch_size x 5*target_size]
        logits_flat = tf.contrib.layers.linear(h, 5*target_size)
        # [batch_size x 5 x target_size]
        logits = tf.reshape(logits_flat, [-1, 5, target_size])

        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order))

        predict = tf.arg_max(tf.nn.softmax(logits), 2)

        return loss, placeholders, predict


