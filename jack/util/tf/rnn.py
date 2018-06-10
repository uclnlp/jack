# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def birnn_with_projection(size, fused_rnn, inputs, length, share_rnn=False, projection_scope=None):
    projection_initializer = tf.constant_initializer(np.concatenate([np.eye(size), np.eye(size)]))
    with tf.variable_scope("RNN", reuse=share_rnn):
        encoded = fused_birnn(fused_rnn, inputs, sequence_length=length, dtype=tf.float32, time_major=False)[0]
        encoded = tf.concat(encoded, 2)

    projected = tf.layers.dense(encoded, size, kernel_initializer=projection_initializer, name=projection_scope)
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


def fused_birnn(fused_rnn, inputs, sequence_length, initial_state=(None, None), dtype=None, scope=None,
                time_major=False, backward_device=None):
    with tf.variable_scope(scope or "BiRNN"):
        sequence_length = tf.cast(sequence_length, tf.int32)
        if not time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])
        outputs_fw, state_fw = fused_rnn(inputs, sequence_length=sequence_length, initial_state=initial_state[0],
                                         dtype=dtype, scope="FW")

        if backward_device is not None:
            with tf.device(backward_device):
                outputs_bw, state_bw = fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state[1], dtype,
                                                          scope="BW")
        else:
            outputs_bw, state_bw = fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state[1], dtype,
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
    with tf.variable_scope(scope or "reader"):
        cell_fw = tf.contrib.rnn.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )
        cell_bw = tf.contrib.rnn.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        if drop_keep_prob != 1.0:
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                                    cell=cell_fw,
                                    output_keep_prob=drop_keep_prob,
                                    input_keep_prob=drop_keep_prob, seed=1233)
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                                    cell=cell_bw,
                                    output_keep_prob=drop_keep_prob,
                                    input_keep_prob=drop_keep_prob, seed=1233)

        all_states_fw_bw, final_states_fw_bw = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            inputs,
            sequence_length=lengths,
            initial_state_fw=initial_state[0],
            initial_state_bw=initial_state[1],
            dtype=tf.float32
        )

        return all_states_fw_bw, final_states_fw_bw


class SRUFusedRNN(tf.contrib.rnn.FusedRNNCell):
    """Simple Recurrent Unit, very fast.  https://openreview.net/pdf?id=rJBiunlAW"""

    def __init__(self, num_units, f_bias=1.0, r_bias=0.0, with_residual=True):
        self._num_units = num_units
        cell = _SRUUpdateCell(num_units, with_residual)
        self._rnn = tf.contrib.rnn.FusedRNNCellAdaptor(cell, use_dynamic_rnn=True)
        self._constant_bias = [0.0] * self._num_units + [f_bias] * self._num_units
        if with_residual:
            self._constant_bias += [r_bias] * self._num_units

        self._constant_bias = np.array(self._constant_bias, np.float32)
        self._with_residual = with_residual

    def __call__(self, inputs, initial_state=None, dtype=tf.float32, sequence_length=None, scope=None):
        num_gates = 3 if self._with_residual else 2
        transformed = tf.layers.dense(inputs, num_gates * self._num_units,
                                      bias_initializer=tf.constant_initializer(self._constant_bias))

        gates = tf.split(transformed, num_gates, axis=2)
        forget_gate = tf.sigmoid(gates[1])
        transformed_inputs = (1.0 - forget_gate) * gates[0]
        if self._with_residual:
            residual_gate = tf.sigmoid(gates[2])
            inputs *= (1.0 - residual_gate)
            new_inputs = tf.concat([inputs, transformed_inputs, forget_gate, residual_gate], axis=2)
        else:
            new_inputs = tf.concat([transformed_inputs, forget_gate], axis=2)

        return self._rnn(new_inputs, initial_state, dtype, sequence_length, scope)


class _SRUUpdateCell(tf.contrib.rnn.RNNCell):
    """Simple Recurrent Unit, very fast.  https://openreview.net/pdf?id=rJBiunlAW"""

    def __init__(self, num_units, with_residual, activation=None, reuse=None):
        super(_SRUUpdateCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._with_residual = with_residual
        self._activation = activation or tf.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Simple recurrent unit (SRU)."""
        if self._with_residual:
            base_inputs, transformed_inputs, forget_gate, residual_gate = tf.split(inputs, 4, axis=1)
            new_state = forget_gate * state + transformed_inputs
            new_h = residual_gate * self._activation(new_state) + base_inputs
        else:
            transformed_inputs, forget_gate = tf.split(inputs, 2, axis=1)
            new_state = new_h = forget_gate * state + transformed_inputs
        return new_h, new_state
