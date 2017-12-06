import logging

import numpy as np
import tensorflow as tf

from jack.tfutil import attention
from jack.tfutil import rnn
from jack.tfutil.activations import activation_from_string
from jack.tfutil.highway import highway_network

logger = logging.getLogger(__name__)


def encoder(sequence, seq_length, repr_dim=100, module='lstm', num_layers=1, conv_width=3,
            dilations=None, reuse=False, residual=False, attn_type=None, num_attn_heads=1, scaled=False,
            activation=None, with_sentinel=False, with_projection=False, name='encoder', **kwargs):
    if num_layers == 1:
        if module == 'lstm':
            out = bi_lstm(repr_dim, sequence, seq_length, name, reuse, with_projection)
            if activation:
                out = activation_from_string(activation)(out)
        elif module == 'sru':
            with_residual = sequence.get_shape()[2].value == repr_dim
            out = bi_sru(repr_dim, sequence, seq_length, with_residual, name, reuse, with_projection)
            if activation:
                out = activation_from_string(activation)(out)
        elif module == 'gru':
            out = bi_rnn(repr_dim, tf.contrib.rnn.BlockGRUCell(repr_dim), sequence,
                         seq_length, name, reuse, with_projection)
            if activation:
                out = activation_from_string(activation)(out)
        elif module == 'gldr':
            out = gated_linear_dilated_residual_network(
                repr_dim, sequence, dilations, conv_width, name, reuse)
        elif module == 'conv':
            out = convnet(repr_dim, sequence, 1, conv_width, activation_from_string(activation), name, reuse)
        elif module == 'conv_glu':
            out = gated_linear_convnet(repr_dim, sequence, 1, conv_width, name, reuse)
        elif module == 'dense':
            out = tf.layers.dense(sequence, repr_dim, name=name, reuse=reuse)
            if activation:
                out = activation_from_string(activation)(out)
        elif module == 'highway':
            out = highway_network(sequence, num_layers, activation_from_string(activation), name=name, reuse=reuse)
        elif module == 'self_attn':
            outs = []
            for i in range(num_attn_heads):
                attn = self_attention(sequence, seq_length, attn_type, scaled, repr_dim, with_sentinel,
                                      activation_from_string(activation), name + str(i), reuse)
                outs.append(attn)
            out = tf.concat(outs, 2) if num_layers > 1 else outs[0]
        else:
            raise ValueError("Unknown encoder type: %s" % module)
    else:
        out = encoder(sequence, seq_length, repr_dim, module, num_layers - 1, conv_width,
                      dilations, reuse, False, attn_type, num_attn_heads, scaled, activation, with_sentinel,
                      with_projection, name)
        out = encoder(out, seq_length, repr_dim, module, 1, conv_width,
                      dilations, reuse, False, attn_type, num_attn_heads, scaled, activation, with_sentinel,
                      with_projection, name + str(num_layers - 1))
    if residual:
        if out.get_shape()[-1].value != sequence.get_shape()[-1].value:
            logging.error('Residual connection only possible if input to sequence encoder %s of type %s has same '
                          'dimension (%d) as output (%d).' % (name, module, sequence.get_shape()[-1].value,
                                                              out.get_shape()[-1].value))
            raise RuntimeError()
        out += sequence
    return out


# RNN Encoders
def _bi_rnn(size, fused_rnn, sequence, seq_length, name, reuse=False, with_projection=False):
    with tf.variable_scope(name, reuse=reuse):
        output = rnn.fused_birnn(fused_rnn, sequence, seq_length, dtype=tf.float32, scope='rnn')[0]
        output = tf.concat(output, 2)
        if with_projection:
            projection_initializer = tf.constant_initializer(np.concatenate([np.eye(size), np.eye(size)]))
            output = tf.layers.dense(output, size, kernel_initializer=projection_initializer, name='projection')
    return output


def bi_lstm(size, sequence, seq_length, name='bilstm', reuse=False, with_projection=False):
    return _bi_rnn(size, tf.contrib.rnn.LSTMBlockFusedCell(size), sequence, seq_length, name, reuse, with_projection)


def bi_rnn(size, rnn_cell, sequence, seq_length, name='bi_rnn', reuse=False, with_projection=False):
    fused_rnn = tf.contrib.rnn.FusedRNNCellAdaptor(rnn_cell, use_dynamic_rnn=True)
    return _bi_rnn(size, fused_rnn, sequence, seq_length, name, reuse, with_projection)


def bi_sru(size, sequence, seq_length, with_residual=True, name='bi_sru', reuse=False, with_projection=False):
    """Simple Recurrent Unit, very fast.  https://openreview.net/pdf?id=rJBiunlAW."""
    fused_rnn = rnn.SRUFusedRNN(size, with_residual)
    return _bi_rnn(size, fused_rnn, sequence, seq_length, name, reuse, with_projection)


# Convolution Encoders

def convnet(out_size, inputs, num_layers, width=3, activation=tf.nn.relu, name='convnet', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        # dim reduction
        output = inputs
        for i in range(num_layers):
            output = _convolutional_block(output, out_size, width=width, name="conv_%d" % i)
    return output


def _convolutional_block(inputs, out_channels, width=3, name='conv', activation=tf.nn.relu):
    channels = inputs.get_shape()[2].value
    # [filter_height, filter_width, in_channels, out_channels]
    f = tf.get_variable(name + '_filter', [width, channels, out_channels])
    output = tf.nn.conv1d(inputs, f, 1, padding='SAME', name=name)
    return activation(output)


# following implementation of fast encoding in https://openreview.net/pdf?id=HJRV1ZZAW
def _residual_dilated_convolution_block(inputs, dilation=1, width=3, name="dilated_conv"):
    # [filter_height, filter_width, in_channels, out_channels]
    output = inputs
    channels = inputs.get_shape()[2].value
    for i in range(2):
        # [filter_height, filter_width, in_channels, out_channels]
        output = _convolutional_glu_block(output, channels, dilation, width, name=name + '_' + str(i))
    return output + inputs


def _convolutional_glu_block(inputs, out_channels, dilation=1, width=3, name='dilated_conv'):
    channels = inputs.get_shape()[2].value
    # [filter_height, filter_width, in_channels, out_channels]
    f = tf.get_variable(name + '_filter', [1, width, channels, out_channels * 2])
    output = tf.nn.atrous_conv2d(tf.expand_dims(inputs, 1), f, dilation, 'SAME', name=name)
    output = tf.squeeze(output, 1)
    output, gate = tf.split(output, 2, 2)
    output *= tf.sigmoid(gate)
    return output


def gated_linear_dilated_residual_network(out_size, inputs, dilations, width=3, name='gldr_network', reuse=False):
    """Follows https://openreview.net/pdf?id=HJRV1ZZAW.

    Args:
        out_size: size of output
        inputs: input sequence tensor [batch_size, length, size]
        dilations: list, representing (half of the) network depth; each residual dilation block comprises 2 convolutions

    Returns:
        [batch_size, length, out_size] tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        # dim reduction
        output = _convolutional_glu_block(inputs, out_size, name='conv_dim_reduction')
        for i, d in enumerate(dilations):
            output = _residual_dilated_convolution_block(output, d, width, name='dilated_conv_%d' % i)
    return output


def gated_linear_convnet(out_size, inputs, num_layers, width=3, name='gated_linear_convnet', reuse=False):
    """Follows https://openreview.net/pdf?id=HJRV1ZZAW.

    Args:
        out_size: size of output
        inputs: input sequence tensor [batch_size, length, size]
        num_layers: number of conv layers with width

    Returns:
        [batch_size, length, out_size] tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        # dim reduction
        output = inputs
        for i in range(num_layers):
            output = _convolutional_glu_block(output, out_size, width=width, name="conv_%d" % i)
    return output


# Self attention layers
def self_attention(inputs, lengths, attn_type='bilinear', scaled=True, repr_dim=None, activation=None,
                   with_sentinel=False, name='self_attention', reuse=False):
    with tf.variable_scope(name, reuse):
        if attn_type == 'bilinear':
            attn_states = attention.bilinear_attention(inputs, inputs, lengths, scaled, with_sentinel)[2]
        elif attn_type == 'dot':
            attn_states = attention.dot_attention(inputs, inputs, lengths, scaled, with_sentinel)[2]
        elif attn_type == 'diagonal_bilinear':
            attn_states = attention.diagonal_bilinear_attention(inputs, inputs, lengths, scaled, with_sentinel)[2]
        elif attn_type == 'mlp':
            attn_states = attention.mlp_attention(repr_dim, activation, inputs, inputs, lengths, with_sentinel)[2]
        else:
            raise ValueError("Unknown attention type: %s" % attn_type)

    return attn_states


# Multi-scale encoder
def multi_scale_birnn_encoder(size, num_layers, cell, sequence, seq_length, name='multiscale_rnn', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        # first use conv net which is important for scale detection
        inputs = _convolutional_block_glu_block(sequence, size, width=5)

        multi_cell = MultiscaleRNNCell(num_layers, cell, 1.0)
        fused_rnn = tf.contrib.rnn.FusedRNNCellAdaptor(multi_cell, use_dynamic_rnn=True)
        inputs = tf.transpose(inputs, [1, 0, 2])
        outputs_fw, state_fw = fused_rnn(inputs, sequence_length=seq_length, dtype=tf.float32, scope="FW")

        outputs_z = outputs_fw[:, :, -num_layers + 1:]
        outputs_fw = outputs_fw[:, :, :-num_layers + 1]
        outputs_fw = tf.transpose(outputs_fw, [1, 0, 2])
        outputs_fw = tf.reshape(outputs_fw, tf.unstack(tf.shape(outputs_fw)[:2]) + [num_layers, cell.output_size])

        rev_outputs_z = tf.reverse_sequence(outputs_z, seq_length, seq_axis=0, batch_axis=1)
        # shift by one to the right
        rev_outputs_z = tf.concat([rev_outputs_z[1:], tf.ones([1, tf.shape(rev_outputs_z)[1], num_layers - 1])], 0)

        # backward rnn with given structure
        multi_cell = MultiscaleRNNCell(num_layers, cell, 1.0, z_given=True)
        fused_rnn = tf.contrib.rnn.FusedRNNCellAdaptor(multi_cell, use_dynamic_rnn=True)
        inputs = tf.concat([inputs, rev_outputs_z], 2)
        outputs_bw, _ = fused_rnn(inputs, sequence_length=seq_length, dtype=tf.float32, scope="BW")
        outputs_bw = tf.reverse_sequence(outputs_bw, seq_length, seq_axis=0, batch_axis=1)
        outputs_bw = tf.transpose(outputs_bw, [1, 0, 2])
        outputs_bw = tf.reshape(outputs_bw, tf.unstack(tf.shape(outputs_bw)[:2]) + [num_layers, cell.output_size])

        outputs_z = tf.concat([tf.ones([tf.shape(outputs_z)[0], tf.shape(outputs_z)[1], 1]), outputs_z], 2)
        # last should always be considered a one, to finish segments
        outputs_z = tf.reverse_sequence(
            tf.concat([tf.ones([1, tf.shape(outputs_z)[1], num_layers]),
                       tf.reverse_sequence(outputs_z, seq_length - 1, seq_dim=0, batch_dim=1)], 0),
            seq_length, seq_dim=0, batch_dim=1)[:-1, :, :]

    return outputs_fw, outputs_bw, tf.transpose(outputs_z, [1, 0, 2])


class MultiscaleRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_layers, cell, slope, z_given=False):
        """
        Args:
            num_layers: number of layers
            cell: cell
        """
        self._cell = cell
        self._num_layers = num_layers
        self._slope = slope
        self._z_given = z_given

    @property
    def output_size(self):
        if self._z_given:
            return self._cell.output_size * self._num_layers
        else:
            return self._cell.output_size * self._num_layers + self._num_layers - 1

    def zero_state(self, batch_size, dtype):
        return ([self._cell.zero_state(batch_size, dtype) for _ in range(self._num_layers)],
                [tf.zeros([batch_size, self._cell.output_size], dtype) for _ in range(self._num_layers)],
                tf.ones([batch_size, self._num_layers - 1]))

    @property
    def state_size(self):
        return ([self._cell.state_size for _ in range(self._num_layers)],
                [self._cell.output_size for _ in range(self._num_layers)],
                self._num_layers - 1)

    def __call__(self, inputs, state, scope=None):
        state, old_outputs, old_z = state
        old_z = tf.split(old_z, self._num_layers - 1, 1)
        state = list(state)
        with tf.variable_scope(scope or 'MultiscaleRNNCell'):
            if self._z_given:
                new_z = tf.split(inputs[:, -self._num_layers + 1:], self._num_layers - 1, 1)
                inputs = inputs[:, :-self._num_layers + 1]
            else:
                new_z = []
            bottom_up = [inputs]
            top_down = old_outputs
            new_states = []
            for i in range(self._num_layers):
                if i < self._num_layers - 1:
                    inputs = tf.concat([top_down[i + 1], bottom_up[i]], 1)
                    flush = old_z[i]
                else:
                    inputs = bottom_up[i]
                    flush = 1.0
                not_flush = 1.0 - flush
                update, copy = (not_flush * new_z[i - 1], not_flush * (1.0 - new_z[i - 1])) if i > 0 else (
                    not_flush, 0.0)

                # keep old state if not flush
                state[i] = tf.contrib.framework.nest.map_structure(lambda t: not_flush * t, state[i])
                out, new_state = self._cell(inputs, state[i], scope='cell_' + str(i))
                new_state_flat = tf.contrib.framework.nest.flatten(new_state)
                state_flat = tf.contrib.framework.nest.flatten(state[i])
                for j, (new_s, s) in enumerate(zip(new_state_flat, state_flat)):
                    state_flat[j] = (update + flush) * new_s + copy * s
                new_state = tf.contrib.framework.nest.pack_sequence_as(state[i], state_flat)
                out = copy * old_outputs[i] + (1 - copy) * out
                bottom_up.append(out)
                new_states.append(new_state)
                if not self._z_given and i < self._num_layers - 1:
                    z = _straightthrough_gate(tf.concat([inputs, old_outputs[i]], 1), self._slope)
                    new_z.append(z if i == 0 else tf.where(new_z[i - 1] > 0, z, new_z[i - 1]))

        new_z = tf.concat(new_z, 1)
        if self._z_given:
            return tf.concat(bottom_up[1:], 1), (new_states, bottom_up[1:], new_z)
        else:
            bottom_up.append(new_z)
            return tf.concat(bottom_up[1:], 1), (new_states, bottom_up[1:-1], new_z)


def _straightthrough_gate(h, slope):
    tf.contrib.distributions.Rel
    g = tf.layers.dense(h, 1)  # , tf.sigmoid)
    g = tf.maximum(0.0, tf.minimum(1.0, (slope * g + 1.0) / 2.0))
    # bound function with straight through estimator
    g = tf.where(g > 0.5, g + tf.stop_gradient(1.0 - g), g - tf.stop_gradient(g))
    return g
