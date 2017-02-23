import tensorflow as tf


def pair_of_bidirectional_LSTMs(seq1, seq1_lengths, seq2, seq2_lengths,
                                output_size, scope=None, drop_keep_prob=1.0,
                                conditional_encoding=True):
    """Duo of bi-LSTMs over seq1 and seq2 with (optional)conditional encoding.

    Args:
        seq1 (tensor = time x batch x input): The inputs into the first biLSTM
        seq1_lengths (tensor = batch): The lengths of the sequences.
        seq2 (tensor = time x batch x input): The inputs into the second biLSTM
        seq1_lengths (tensor = batch): The lengths of the sequences.
        output_size (int): Size of the LSTMs state.
        scope (string): The TensorFlow scope for the reader.
        drop_keep_drop (float=1.0): The keep propability for dropout.

    Returns:
        Outputs (tensor): The outputs from the second bi-LSTM.
        States (tensor): The cell states from the second bi-LSTM.
    """
    with tf.variable_scope(scope or "paired_LSTM_seq1") as varscope1:
        # seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        _, seq1_final_states = dynamic_bidirectional_lstm(
                        seq1, seq1_lengths, output_size, scope=varscope1,
                        drop_keep_prob=drop_keep_prob)

    with tf.variable_scope(scope or "paired_LSTM_seq2") as varscope2:
        varscope1.reuse_variables()
        # each [batch_size x max_seq_length x output_size]
        all_states_fw_bw, final_states_fw_bw = dynamic_bidirectional_lstm(
                                            seq2, seq2_lengths, output_size,
                                            seq1_final_states, scope=varscope2,
                                            drop_keep_prob=drop_keep_prob)

    return all_states_fw_bw, final_states_fw_bw


def dynamic_bidirectional_lstm(inputs, lengths, output_size,
                               initial_state=(None, None), scope=None,
                               drop_keep_prob=1.0):
    """Dynamic bi-LSTM reader, with optional initial state.

    Args:
        inputs (tensor): The inputs into the bi-LSTM
        lengths (tensor): The lengths of the sequences
        output_size (int): Size of the LSTM state of the reader.
        context (tensor=None, tensor=None): Tuple of initial
                                            (forward, backward) states
                                            for the LSTM
        scope (string): The TensorFlow scope for the reader.
        drop_keep_drop (float=1.0): The keep probability for dropout.

    Returns:
        all_states (tensor): All forward and backward states
        final_states (tensor): The final forward and backward states
    """
    with tf.variable_scope(scope or "reader") as varscope:
        varscope
        cell = tf.contrib.rnn.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        if drop_keep_prob != 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(
                                    cell=cell,
                                    output_keep_prob=drop_keep_prob,
                                    input_keep_prob=drop_keep_prob, seed=1233)

        all_states_fw_bw, final_states_fw_bw = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            inputs,
            sequence_length=lengths,
            initial_state_fw=initial_state[0],
            initial_state_bw=initial_state[1],
            dtype=tf.float32
        )

        return all_states_fw_bw, final_states_fw_bw


def fully_connected_projection(inputs, output_size):
    """Projects inputs onto target dimension. Returns logits, loss, and argmax.

    Creates fully connected projection layer. Then applies cross entropy
    softmax to get the loss. Calculate predictions via argmax.
    Args:
        inputs (tensor): Input into the projection layer.
        output_size (int): Size of the targets (used in projection layer).
    """
    init = tf.contrib.layers.xavier_initializer(uniform=True) #uniform=False for truncated normal
    logits = tf.contrib.layers.fully_connected(inputs, output_size, weights_initializer=init, activation_fn=None)
    return logits
