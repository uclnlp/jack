import tensorflow as tf
import numpy as np


def birnn_with_projection(size, fused_rnn_constructor, inputs, length, share_rnn=False, projection_scope=None):
    projection_initializer = tf.constant_initializer(np.concatenate([np.eye(size), np.eye(size)]))
    fused_rnn = fused_rnn_constructor(size)
    with tf.variable_scope("RNN", reuse=share_rnn):
        encoded = fused_birnn(fused_rnn, inputs, sequence_length=length, dtype=tf.float32, time_major=False)[0]
        encoded = tf.concat(2, encoded)

    projected = tf.contrib.layers.fully_connected(encoded, size,
                                                  activation_fn=tf.tanh,
                                                  weights_initializer=projection_initializer,
                                                  scope=projection_scope)
    return projected


def fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state=None, dtype=None, scope=None, time_major=True):
    if not time_major:
        inputs = tf.transpose(inputs, [1, 0, 2])
    #assumes that time dim is 0 and batch is 1
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
                outputs_bw, state_bw = fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state, dtype, scope="BW")
        else:
            outputs_bw, state_bw = fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state, dtype, scope="BW")

        if not time_major:
            outputs_fw = tf.transpose(outputs_fw, [1, 0, 2])
            outputs_bw = tf.transpose(outputs_bw, [1, 0, 2])
    return (outputs_fw, outputs_bw), (state_fw, state_bw)