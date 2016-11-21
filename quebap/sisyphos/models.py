import tensorflow as tf
import numpy as np

def embedder(inputs, input_size, vocab_size, embeddings=None, scope=None):
    """
    embeds the given input indices into a [batch_size, max_seq_length, input_size] tensor, based on the
    tensor embeddings [vocab_size, embedding_size]. Creates the embeddings tensor if needed.
    In case input_size differs from embedding_size, a linear transformation is performed.
    :param inputs: tensor with instance indices for the current batch (typically a placeholder)
    :param input_size: embedding length
    :param vocab_size: number of symbols in the vocabulary
    :param embeddings: tensor with shape [vocab_size, embedding_size], or None.
    :param scope: scope for the embedding matrix and embedded inputs
    :return: [batch_size, max_seq_length, input_size] tensor with embedded inputs
    """

    # todo: clean up - input arguments input_size, vocab_size not needed if embeddings is not None
    with tf.variable_scope(scope or "embedder") as varscope:

        # initializer = tf.random_normal_initializer(0.0, 0.1)
        initializer = tf.random_uniform_initializer(-0.05, 0.05)
        if embeddings is None:
            embeddings = \
                tf.get_variable("W", [vocab_size, input_size],
                                initializer=initializer)
            # [batch_size, max_seq_length, input_size]
            return tf.nn.embedding_lookup(embeddings, inputs)

        else:
            static_shape = tf.Tensor.get_shape(embeddings)
            if static_shape[1] == input_size:
                # [batch_size, max_seq_length, input_size]
                return tf.nn.embedding_lookup(embeddings, inputs)
            else:
                lin = tf.get_variable("W_trf", [1,static_shape[1], input_size],
                                    initializer=initializer)
                embedded = tf.nn.embedding_lookup(embeddings, inputs)
                lins = tf.tile(lin,[tf.shape(embedded)[0],1,1])
                rescaled_embedded = tf.batch_matmul(embedded,lins)
                #restore some shape information
                rescaled_embedded.set_shape([None, None, input_size])
                return rescaled_embedded






def create_embeddings(vocab, retrain=False, scope=None):
    """
    create embedding tensor with pre-trained embeddings
    :param vocab: instance of class VocabEmb
    :param retrain: False if pretrained embeddings are fixed, True otherwise
    :return: embeddings tensor with shape [vocab_size, embedding_length]
    """
    # todo: additional functionality to extend pretrained embeddings with non/trainable extra dimensions

    assert vocab.__class__.__name__=="VocabEmb", 'create_embeddings() needs VocabEmb instance'
    # todo: reduce VocabEmb to single Vocab class

    #embeddings = np.random.normal(loc=0.0, scale=0.1, size=[len(vocab), input_size]).astype("float32")

    v_shape = vocab.get_shape()
    embeddings = np.random.uniform(low=-0.05, high=0.05, size=[v_shape[0], v_shape[1]]).astype("float32")
    index_pretrained = vocab.get_normalized_ids_pretrained()
    syms_pretrained = vocab.get_syms_pretrained()
    for i,sym in zip(index_pretrained,syms_pretrained):
        vec = vocab.emb(sym)
        if vec is not None: #should not happen if same emb was used to create vocab
            embeddings[i] = vec

    with tf.variable_scope(scope or "embedder") as varscope:
        if retrain: #all are trainable
            E = tf.get_variable("W", initializer=tf.identity(embeddings), trainable=True)
        else:
            E_tune = tf.get_variable("W_tune", initializer=tf.identity(embeddings[:vocab.count_oov()]), trainable=True)
            E_fixed = tf.get_variable("W_fixed", initializer=tf.identity(embeddings[vocab.count_oov():]), trainable=False)
            E = tf.concat(0, [E_tune, E_fixed], name="W")
            #first out-of-vocab (tunable), then pre-trained, corresponding to normalized VocabEmb indices
    return E


def get_total_trainable_variables():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def get_total_variables():
    total_parameters = 0
    for variable in tf.all_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def reader(inputs, lengths, output_size, contexts=(None, None), scope=None):
    with tf.variable_scope(scope or "reader") as varscope:
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            inputs,
            sequence_length=lengths,
            initial_state_fw=contexts[0],
            initial_state_bw=contexts[1],
            dtype=tf.float32
        )

        # ( (outputs_fw,outputs_bw) , (output_state_fw,output_state_bw) )
        # in case LSTMCell: output_state_fw = (c_fw,h_fw), and output_state_bw = (c_bw,h_bw)
        # each [batch_size x max_seq_length x output_size]
        return outputs, states


def conditional_reader(seq1, seq1_lengths, seq2, seq2_lengths, output_size, scope=None):
    with tf.variable_scope(scope or "conditional_reader_seq1") as varscope1:
        #seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        _, seq1_states = reader(seq1, seq1_lengths, output_size, scope=varscope1)
    with tf.variable_scope(scope or "conditional_reader_seq2") as varscope2:
        varscope1.reuse_variables()
        # each [batch_size x max_seq_length x output_size]
        return reader(seq2, seq2_lengths, output_size, seq1_states, scope=varscope1)


def bag_reader(inputs, lengths):
        output=tf.reduce_sum(inputs,1,keep_dims=False)
        return output
        

def boe_reader(seq1, seq1_lengths, seq2, seq2_lengths):
        output1 = bag_reader(seq1, seq1_lengths)
        output2 = bag_reader(seq2, seq2_lengths)
        # each [batch_size x max_seq_length x output_size]
        return tf.concat(1,[output1,output2])


def predictor(output, targets, target_size):
    logits = tf.contrib.layers.fully_connected(output, target_size)
    loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets))
    predict = tf.arg_max(tf.nn.softmax(logits), 1)
    return logits, loss, predict


def conditional_reader_model(input_size, output_size, vocab_size, target_size,
                             embeddings=None, attentive=False):
    # Model
    # [batch_size, max_seq1_length]
    sentence1 = tf.placeholder(tf.int64, [None, None], "sentence1")
    # [batch_size]
    sentence1_lengths = tf.placeholder(tf.int64, [None], "sentence1_lengths")

    # [batch_size, max_seq2_length]
    sentence2 = tf.placeholder(tf.int64, [None, None], "sentence2")
    # [batch_size]
    sentence2_lengths = tf.placeholder(tf.int64, [None], "sentence2_lengths")

    # [batch_size]
    targets = tf.placeholder(tf.int64, [None], "targets")

    with tf.variable_scope("embedders") as varscope:
        seq1_embedded = embedder(sentence1, input_size, vocab_size, embeddings=embeddings)
        varscope.reuse_variables()
        seq2_embedded = embedder(sentence2, input_size, vocab_size, embeddings=embeddings)


    print('TRAINABLE VARIABLES (only embeddings): %d'%get_total_trainable_variables())

    if attentive:
        le_conditional_reader = conditional_attentive_reader
    else:
        le_conditional_reader = conditional_reader

    outputs, states = le_conditional_reader(seq1_embedded, sentence1_lengths,
                                            seq2_embedded, sentence2_lengths,
                                            output_size)

    # fixme: also use bidirectional encoding for attention model?
    if isinstance(states, tf.nn.rnn_cell.LSTMStateTuple):
        output = states.h
    elif isinstance(states, tuple):
        # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
        output = tf.concat(1, [states[0][1], states[1][1]])
    else:
        raise AttributeError

    logits, loss, predict = predictor(output, targets, target_size)

    print('TRAINABLE VARIABLES (embeddings + model): %d'%get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d'%get_total_variables())


    return (logits, loss, predict), \
           {'sentence1': sentence1, 'sentence1_lengths': sentence1_lengths,
            'sentence2': sentence2, 'sentence2_lengths': sentence2_lengths,
            'targets': targets} #placeholders


def conditional_attentive_reader(seq1, seq1_lengths, seq2, seq2_lengths,
                                 output_size, scope=None):
    with tf.variable_scope(scope or "conditional_attentive_reader_seq1") as varscope1:
        #seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        attention_states, seq1_states = reader(seq1, seq1_lengths, output_size, scope=varscope1)
    with tf.variable_scope(scope or "conditional_attentitve_reader_seq2") as varscope2:
        varscope1.reuse_variables()

        batch_size = tf.shape(seq2)[0]
        max_time = tf.shape(seq2)[1]
        input_depth = int(seq2.get_shape()[2])

        #batch_size = tf.Print(batch_size, [batch_size])
        #max_time = tf.Print(max_time, [max_time])

        # transforming seq2 to time major
        seq2 = tf.transpose(seq2, [1, 0, 2])
        num_units = output_size

        # fixme: very hacky and costly way
        seq2_lengths = tf.cast(seq2_lengths, tf.int32)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unpack(seq2)

        cell = tf.nn.rnn_cell.LSTMCell(num_units)

        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:  # time == 0
                next_cell_state = cell.zero_state(batch_size, tf.float32)
            else:
                next_cell_state = cell_state
            elements_finished = (time >= seq2_lengths)

            c, query = next_cell_state

            # [max_time x batch_size x num_units]
            query_expanded = tf.tile(tf.expand_dims(query, 0), [max_time, 1, 1])

            attention_states_projected = \
                tf.contrib.layers.linear(attention_states, num_units)
            query_projected = \
                tf.contrib.layers.linear(query_expanded, num_units)

            # [max_time x batch_size x num_units]
            M = tf.tanh(attention_states_projected + query_projected)

            # [batch_size x max_time]
            logits = tf.transpose(tf.squeeze(tf.contrib.layers.linear(M, 1)))

            # [max_time x batch_size]
            alpha = tf.transpose(tf.nn.softmax(logits))

            attention_states_flat = tf.reshape(attention_states,
                                               [-1, num_units])
            alpha_flat = tf.reshape(alpha, [-1, 1])

            # [batch_size x num_units]
            r = attention_states_flat * alpha_flat
            r_reshaped = tf.reduce_sum(
                tf.reshape(r, [max_time, batch_size, num_units]), [0])

            # [batch_size x num_units]
            h = tf.tanh(tf.contrib.layers.linear(
                tf.concat(1, [query, r_reshaped]), num_units))

            next_cell_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
                lambda: inputs_ta.read(time))
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        outputs = outputs_ta.pack()

        outputs_batch_major = tf.transpose(outputs, [1, 0, 2])

        # each [batch_size x max_seq_length x output_size]
        return outputs_batch_major, final_state

if __name__ == '__main__':
    max_time = 3
    batch_size = 2
    input_depth = 5
    num_units = 7

    # todo: later this will be the output of another reader
    attention_states = tf.placeholder(shape=(max_time, batch_size, num_units),
                                      dtype=tf.float32)

    inputs = tf.placeholder(shape=(max_time, batch_size, input_depth),
                            dtype=tf.float32)
    sequence_length = tf.placeholder(shape=(batch_size,), dtype=tf.int32)

    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unpack(inputs)

    cell = tf.nn.rnn_cell.LSTMCell(num_units)

    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:  # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
        else:
            next_cell_state = cell_state
        elements_finished = (time >= sequence_length)

        c, query = next_cell_state

        # [max_time x batch_size x num_units]
        query_expanded = tf.tile(tf.expand_dims(query, 0), [max_time, 1, 1])

        attention_states_projected = \
            tf.contrib.layers.linear(attention_states, num_units)
        query_projected = \
            tf.contrib.layers.linear(query_expanded, num_units)

        # [max_time x batch_size x num_units]
        M = tf.tanh(attention_states_projected + query_projected)

        # [batch_size x max_time]
        logits = tf.transpose(tf.squeeze(tf.contrib.layers.linear(M, 1)))

        # [max_time x batch_size]
        alpha = tf.transpose(tf.nn.softmax(logits))

        attention_states_flat = tf.reshape(attention_states, [-1, num_units])
        alpha_flat = tf.reshape(alpha, [-1, 1])

        # [batch_size x num_units]
        r = attention_states_flat * alpha_flat
        r_reshaped = tf.reduce_sum(
            tf.reshape(r, [max_time, batch_size, num_units]), [0])

        # [batch_size x num_units]
        h = tf.tanh(tf.contrib.layers.linear(
            tf.concat(1, [query, r_reshaped]), num_units))

        next_cell_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        finished = tf.reduce_all(elements_finished)
        next_input = tf.cond(
            finished,
            lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
            lambda: inputs_ta.read(time))
        next_loop_state = None
        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)


    outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
    outputs = outputs_ta.pack()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        feed_dict = {
            attention_states: np.random.randn(max_time, batch_size, num_units),
            inputs: np.random.randn(max_time, batch_size, input_depth),
            sequence_length: np.random.randint(1, max_time, batch_size)
        }

        print(feed_dict[sequence_length])
        results = sess.run(outputs, feed_dict)
        print(results)

