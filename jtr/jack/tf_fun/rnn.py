import numpy as np
import tensorflow as tf


def birnn_with_projection(size, fused_rnn_constructor, inputs, length, share_rnn=False, projection_scope=None):
    projection_initializer = tf.constant_initializer(np.concatenate([np.eye(size), np.eye(size)]))
    fused_rnn = fused_rnn_constructor(size)
    with tf.variable_scope("RNN", reuse=share_rnn):
        encoded = fused_birnn(fused_rnn, inputs, sequence_length=length, dtype=tf.float32, time_major=False)[0]
        encoded = tf.concat(encoded, 2)

    projected = tf.contrib.layers.fully_connected(encoded, size, activation_fn=None,
                                                  weights_initializer=projection_initializer,
                                                  scope=projection_scope)
    return projected


def fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state=None, dtype=None, scope=None, time_major=True):
    if not time_major:
        inputs = tf.transpose(inputs, [1, 0, 2])
    # assumes that time dim is 0 and batch is 1
    rev_inputs = tf.reverse_sequence(inputs, sequence_length, 0, 1)
    rev_outputs, last_state = fused_rnn(rev_inputs, sequence_length=sequence_length, initial_state=initial_state,
                                        dtype=dtype, scope=scope)
    outputs = tf.reverse_sequence(rev_outputs, sequence_length, 0, 1)
    if not time_major:
        outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs, last_state


def fused_birnn(fused_rnn, inputs, sequence_length, initial_state=None, dtype=None, scope=None, time_major=True,
                backward_device=None):
    with tf.variable_scope(scope or "BiRNN"):
        sequence_length = tf.cast(sequence_length, tf.int32)
        if not time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])
        outputs_fw, state_fw = fused_rnn(inputs, sequence_length=sequence_length, initial_state=initial_state,
                                         dtype=dtype, scope="FW")

        if backward_device is not None:
            with tf.device(backward_device):
                outputs_bw, state_bw = fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state, dtype,
                                                          scope="BW")
        else:
            outputs_bw, state_bw = fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state, dtype,
                                                      scope="BW")

        if not time_major:
            outputs_fw = tf.transpose(outputs_fw, [1, 0, 2])
            outputs_bw = tf.transpose(outputs_bw, [1, 0, 2])
    return (outputs_fw, outputs_bw), (state_fw, state_bw)




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
