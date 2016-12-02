import tensorflow as tf
import pprint

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

        # fixme: this is only using the BW state!
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

        fw = states[0][1]

        # todo: also use backward pass
        # bw = states[1][1]

        h = fw

        # [batch_size x 5*target_size]
        logits_flat = tf.contrib.layers.linear(h, 5*target_size)
        # [batch_size x 5 x target_size]
        logits = tf.reshape(logits_flat, [-1, 5, target_size])

        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order))

        predict = tf.arg_max(tf.nn.softmax(logits), 2)

        return loss, placeholders, predict


def get_selective_model(vocab_size, input_size, output_size, target_size, layers=1,
                        dropout=0.0, nvocab=None):
    # Placeholders
    # [batch_size x 5 x max_length]
    # fixme: or [5 x batch_size x max_length]?
    story = tf.placeholder(tf.int64, [None, None, None], "story")
    # [batch_size x 5]
    story_length = tf.placeholder(tf.int64, [None, None], "story_length")
    # [batch_size x 5]
    order = tf.placeholder(tf.int64, [None, None], "order")
    placeholders = {"story": story, "story_length": story_length,
                    "order": order}

    batch_size = tf.shape(story)[0]

    # 5 times [batch_size x max_length]
    sentences = [tf.reshape(x, [batch_size, -1]) for x in tf.split(1, 5, story)]

    # 5 times [batch_size]
    lengths = [tf.reshape(x, [batch_size])
               for x in tf.split(1, 5, story_length)]

    # Word embeddings
    if nvocab is None:
        initializer = tf.random_uniform_initializer(-0.05, 0.05)
        embeddings = tf.get_variable("W", [vocab_size, input_size],
                                     initializer=initializer)
    else:
        print('..using pretrained embeddings')
        embeddings = nvocab.embedding_matrix

    # [batch_size x max_seq_length x input_size]
    sentences_embedded = [tf.nn.embedding_lookup(embeddings, sentence)
                          for sentence in sentences]

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

        with tf.variable_scope("rnn") as rnn_varscope:
            # 5 times outputs, states
            rnn_result = []
            for i, (sentence, length) in \
                    enumerate(zip(sentences_embedded, lengths)):
                if i > 0:
                    rnn_varscope.reuse_variables()

                rnn_result.append(
                    tf.nn.bidirectional_dynamic_rnn(
                        cell_dropout,
                        cell_dropout,
                        sentence,
                        sequence_length=length,
                        dtype=tf.float32
                    )
                )

        fws = [states[1][0][1] for states in rnn_result]

        # todo: also use backward pass
        # bws = [states[1][1][1] for states in rnn_result]

        # 5 times [batch_size x target_size]
        hs = fws

        # [batch_size x 5*output_size]
        h = tf.concat(1, hs)

        # [batch_size x 5*target_size]
        logits_flat = tf.contrib.layers.linear(h, 5*target_size)
        # [batch_size x 5 x target_size]
        logits = tf.reshape(logits_flat, [-1, 5, target_size])

        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order))

        predict = tf.arg_max(tf.nn.softmax(logits), 2)

        return loss, placeholders, predict
