import tensorflow as tf
import numpy as np





def get_total_trainable_variables():
    """Calculates and returns the number of trainable parameters in the model."""
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def get_total_variables():
    """Calculates and returns the number of parameters in the model (these can be fixed)."""
    total_parameters = 0
    for variable in tf.all_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def reader(inputs, lengths, output_size, contexts=(None, None), scope=None, drop_keep_prob=1.0):
    """Dynamic bi-LSTM reader; can be conditioned with initial state of other rnn.

    Args:
        inputs (tensor): The inputs into the bi-LSTM
        lengths (tensor): The lengths of the sequences
        output_size (int): Size of the LSTM state of the reader.
        context (tensor=None, tensor=None): Tuple of initial (forward, backward) states
                                  for the LSTM
        scope (string): The TensorFlow scope for the reader.
        drop_keep_drop (float=1.0): The keep probability for dropout.

    Returns:
        Outputs (tensor): The outputs from the bi-LSTM.
        States (tensor): The cell states from the bi-LSTM.
    """
    with tf.variable_scope(scope or "reader") as varscope:
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        if drop_keep_prob != 1.0:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=drop_keep_prob)

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


def conditional_reader(seq1, seq1_lengths, seq2, seq2_lengths, output_size, scope=None, drop_keep_prob=1.0):
    """Duo of bi-LSTMs over seq1 and seq2 where seq2 is conditioned on by the final state of seq1.

    See also: conditional_reader_model: Instantiates either condition or
    attentive reader with placeholders.
    Args:
        seq1 (tensor): The inputs into the first bi-LSTM.
        seq1_lengths (tensor): The lengths of the sequences.
        seq2 (tensor): The inputs into the second bi-LSTM.
        seq1_lengths (tensor): The lengths of the sequences.
        output_size (int): Size of the LSTMs state.
        scope (string): The TensorFlow scope for the reader.
        drop_keep_drop (float=1.0): The keep propability for dropout.

    Returns:
        Outputs (tensor): The outputs from the second bi-LSTM.
        States (tensor): The cell states from the second bi-LSTM.
    """
    with tf.variable_scope(scope or "conditional_reader_seq1") as varscope1:
        #seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        _, seq1_states = reader(seq1, seq1_lengths, output_size, scope=varscope1, drop_keep_prob=drop_keep_prob)
    with tf.variable_scope(scope or "conditional_reader_seq2") as varscope2:
        varscope1.reuse_variables()
        # each [batch_size x max_seq_length x output_size]
        return reader(seq2, seq2_lengths, output_size, seq1_states, scope=varscope2, drop_keep_prob=drop_keep_prob)


def bilstm_readers(seq1, seq1_lengths, seq2, seq2_lengths, output_size, scope=None, drop_keep_prob=1.0):
    """Duo of independent bi-LSTMs over seq1 and seq2.

    Args:
        seq1 (tensor): The inputs into the first bi-LSTM.
        seq1_lengths (tensor): The lengths of the sequences.
        seq2 (tensor): The inputs into the second bi-LSTM.
        seq1_lengths (tensor): The lengths of the sequences.
        output_size (int): Size of the LSTMs state.
        scope (string): The TensorFlow scope for the reader.
        drop_keep_drop (float=1.0): The keep probability for dropout.

    Returns:
        Outputs (tensor): The outputs from the second bi-LSTM.
        States (tensor): The cell states from the second bi-LSTM.
    """
    # same as conditional_reader, apart from that second lstm is initialised randomly
    with tf.variable_scope(scope or "bilstm_reader_seq1") as varscope1:
        # seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        seq1_output, seq1_states = reader(seq1, seq1_lengths, output_size, scope=varscope1, drop_keep_prob=drop_keep_prob)
    with tf.variable_scope(scope or "bilstm_reader_seq2") as varscope2:
        varscope1.reuse_variables()
        # each [batch_size x max_seq_length x output_size]
        seq2_output, seq2_states = reader(seq2, seq2_lengths, output_size, scope=varscope2, drop_keep_prob=drop_keep_prob)
    return seq1_output, seq1_states, seq2_output, seq2_states


def bag_reader(inputs, lengths):
    """Sums along the feature dimension of inputs. Returns sum per sample.

    Args:
        inputs (tensor): Input sequence.
        lengths (tensor):  Variable lengths of the input sequence.
    Returns:
        output (tensor: Sum per sample.
    """
    output=tf.reduce_sum(inputs,1,keep_dims=False)
    return output

def boe_reader(seq1, seq1_lengths, seq2, seq2_lengths):
    """Sums the feature dimension of two sequences and return its concatenation

    Args:
        seq1 (tensor): First input sequence.
        seq1_lengths (tensor):  Variable lengths of the first input sequence.
        seq2 (tensor): Second input sequence.
        seq2_lengths (tensor):  Variable lengths of the second input sequence.
    Returns:
        output (tensor: Concatenation of the sums per sample for the sequences.
    """
    output1 = bag_reader(seq1, seq1_lengths)
    output2 = bag_reader(seq2, seq2_lengths)
    # each [batch_size x max_seq_length x output_size]
    return tf.concat(1,[output1,output2])


def predictor(inputs, targets, target_size):
    """Projects inputs onto targets. Returns logits, loss, argmax.

    Creates fully connected projection layer(logits). Then applies cross entropy
    softmax to get the loss. Calculate predictions via argmax.
    Args:
        inputs (tensor): Input into the projection layer.
        targets (tensor): Targets for the loss function.
        target_size (int): Size of the targets (used in projection layer).
    """
    init = tf.contrib.layers.xavier_initializer(uniform=True) #uniform=False for truncated normal
    logits = tf.contrib.layers.fully_connected(inputs, target_size, weights_initializer=init, activation_fn=None)
    #note: standard relu applied; switch off at last layer with activation_fn=None!

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')
    return logits, loss, predict


def conditional_reader_model(output_size, target_size, nvocab, attentive=False):
    """Creates reader that is attentive (each step) or conditional (last step)

    Reference: Teaching Machines to Read and Comprehend
    Link: https://arxiv.org/pdf/1506.03340.pdf

    Creates either a conditional reader (condition on last timestep only) or
    attentive reader (condition on all timesteps).
    The model consists of two bi-directional LSTM where the second LSTM
    is conditioned by the last cell state (conditional) or by an attention
    value from all past cell states.

    Args:
        output_size (int): Size of the two bi-LSTMs.
        target_size (int): Size of the targets (number of labels).
        nvocab (NeuralVocab): NeuralVocab class; maps word-ids to embeddings.
        attentive (bool=False): To create the attentive reader or not.
    Returns:
        (logits (tensor), loss (tensor), predict (tensor): Triple of logits,
        loss and predict (argmax) tensors,
       {'sentence1': sentence1 (TensorFlow placeholder),
        'sentence1_lengths': sentence1_lengths (tensor),
        'sentence2': sentence2 (TensorFlow placeholder),
        'sentence2_lengths': sentence2_lengths (tensor),
        'targets': targets (tensor)}: Dictionary of placeholders to feed data
        into.
    """
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
        seq1_embedded = nvocab(sentence1)
        varscope.reuse_variables()
        seq2_embedded = nvocab(sentence2)
#        seq1_embedded = embedder(sentence1, input_size, vocab_size, embeddings=embeddings)
#        seq2_embedded = embedder(sentence2, input_size, vocab_size, embeddings=embeddings)


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
    """Creates attentive reader where two bi-LSTMs attend to each others state.

    Reference: Teaching Machines to Read and Comprehend
    Link: https://arxiv.org/pdf/1506.03340.pdf

    Creates an attentive reader which consists of two bi-directional LSTM
    where the second LSTM is conditioned by an attention value from all
    past cell states.

    Args:
        `seq1 (tensor)`: Size of first input sequence.
        `seq1_lengths (tensor)`: Lengths of first input sequences.
        seq2 (tensor): Size of second input sequence.
        seq2_lengths (tensor): Lengths of second input sequences.
        output_size (int): Hidden unit size of the two bi-LSTMs. 
        scope (string): The TensorFlow scope for the reader.
    Returns:
        (logits (tensor), loss (tensor), predict (tensor): Triple of logits,
        loss and predict (argmax) tensors,
       {'sentence1': sentence1 (TensorFlow placeholder),
        'sentence1_lengths': sentence1_lengths (tensor),
        'sentence2': sentence2 (TensorFlow placeholder),
        'sentence2_lengths': sentence2_lengths (tensor),
        'targets': targets (tensor)}: Dictionary of placeholders to feed data
        into.
    """
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

        attention_states_fw, attention_states_bw = tf.split(0, 2, attention_states)
        attention_states = tf.concat(3, [attention_states_fw, attention_states_bw])
        attention_states = tf.squeeze(attention_states, [0])
        # transforming attention states time major
        attention_states = tf.transpose(attention_states, [1, 0, 2])

        attention_states = tf.contrib.layers.linear(attention_states, num_units)

        att_len = tf.shape(attention_states)[0]

        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:  # time == 0
                next_cell_state = cell.zero_state(batch_size, tf.float32)
            else:
                next_cell_state = cell_state
            elements_finished = (time >= seq2_lengths)

            c, query = next_cell_state

            # [att_len x batch_size x num_units]
            query_expanded = tf.tile(tf.expand_dims(query, 0), [att_len, 1, 1])

            attention_states_projected = \
                tf.contrib.layers.linear(attention_states, num_units)

            query_projected = \
                tf.contrib.layers.linear(query_expanded, num_units)

            # [att_len x batch_size x num_units]
            M = tf.tanh(attention_states_projected + query_projected)

            # [batch_size x att_len]
            logits = tf.transpose(tf.squeeze(tf.contrib.layers.linear(M, 1)))

            # [att_len x batch_size]
            alpha = tf.transpose(tf.nn.softmax(logits))

            attention_states_flat = tf.reshape(attention_states, [-1, num_units])

            alpha_flat = tf.reshape(alpha, [-1, 1])

            # [batch_size x num_units]
            r = attention_states_flat * alpha_flat
            r_reshaped = tf.reduce_sum(
                tf.reshape(r, [att_len, batch_size, num_units]), [0])

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
