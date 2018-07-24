import logging

import numpy as np
import tensorflow as tf

from jack.util.tf import attention, rnn
from jack.util.tf.activations import activation_from_string
from jack.util.tf.highway import highway_network

logger = logging.getLogger(__name__)


def encoder(sequence, seq_length, repr_dim=100, module='lstm', num_layers=1, reuse=None, residual=False,
            activation=None, layer_norm=False, name='encoder', dropout=None, is_eval=True, **kwargs):
    if num_layers == 1:
        if layer_norm:
            with tf.variable_scope('layernorm', reuse=False) as vs:
                vs._reuse = False  # HACK
                num_layernorms = sum(1 for v in vs.global_variables() if 'layernorm' in v.name)
                sequence = tf.contrib.layers.layer_norm(sequence, scope=str(num_layernorms))

        with tf.variable_scope(name, reuse=reuse):
            if module == 'lstm':
                out = bi_lstm(repr_dim, sequence, seq_length, **kwargs)
                if activation:
                    out = activation_from_string(activation)(out)
            elif module == 'sru':
                with_residual = sequence.get_shape()[2].value == repr_dim
                out = bi_sru(repr_dim, sequence, seq_length, with_residual, **kwargs)
                if activation:
                    out = activation_from_string(activation)(out)
            elif module == 'rnn':
                out = bi_rnn(repr_dim, tf.nn.rnn_cell.BasicRNNCell(repr_dim, activation_from_string(activation)),
                             sequence, seq_length, **kwargs)
            elif module == 'gru':
                out = bi_rnn(repr_dim, tf.contrib.rnn.GRUBlockCell(repr_dim), sequence,
                             seq_length, **kwargs)
                if activation:
                    out = activation_from_string(activation)(out)
            elif module == 'gldr':
                out = gated_linear_dilated_residual_network(
                    repr_dim, sequence, **kwargs)
            elif module == 'conv':
                out = convnet(repr_dim, sequence, 1, activation=activation_from_string(activation), **kwargs)
            elif module == 'conv_glu':
                out = gated_linear_convnet(repr_dim, sequence, 1, **kwargs)
            elif module == 'conv_separable':
                out = depthwise_separable_convolution(
                    repr_dim, sequence, activation=activation_from_string(activation), **kwargs)
            elif module == 'dense':
                out = tf.layers.dense(sequence, repr_dim)
                if activation:
                    out = activation_from_string(activation)(out)
            elif module == 'highway':
                out = highway_network(sequence, num_layers, activation_from_string(activation))
            elif module == 'self_attn':
                outs = []
                num_attn = kwargs.get('num_attn_heads', 1)
                for i in range(num_attn):
                    with tf.variable_scope(str(i)):
                        attn = self_attention(sequence, seq_length, repr_dim=repr_dim, **kwargs)
                        outs.append(attn)
                out = tf.concat(outs, 2) if num_attn > 1 else outs[0]
            elif module == 'positional_encoding':
                out = positional_encoding(sequence, seq_length)
            else:
                raise ValueError("Unknown encoder type: %s" % module)

            if residual:
                if out.get_shape()[-1].value != sequence.get_shape()[-1].value:
                    logging.error(
                        'Residual connection only possible if input to sequence encoder %s of type %s has same '
                        'dimension (%d) as output (%d).' % (name, module, sequence.get_shape()[-1].value,
                                                            out.get_shape()[-1].value))
                    raise RuntimeError()
                out += sequence

            if dropout is not None:
                out = tf.cond(
                    tf.logical_and(tf.greater(dropout, 0.0), tf.logical_not(is_eval)),
                    lambda: tf.nn.dropout(out, 1.0 - dropout, noise_shape=[tf.shape(out)[0], 1, tf.shape(out)[-1]]),
                    lambda: out)
    else:
        out = encoder(sequence, seq_length, repr_dim, module, num_layers - 1, reuse, residual,
                      activation, layer_norm, name, dropout=dropout, is_eval=is_eval, **kwargs)

        out = encoder(out, seq_length, repr_dim, module, 1, reuse, residual, activation, layer_norm,
                      name + str(num_layers - 1), dropout=dropout, is_eval=is_eval, **kwargs)

    return out


# RNN Encoders
def _bi_rnn(size, fused_rnn, sequence, seq_length, with_projection=False):
    output = rnn.fused_birnn(fused_rnn, sequence, seq_length, dtype=tf.float32, scope='rnn')[0]
    output = tf.concat(output, 2)
    if with_projection:
        projection_initializer = tf.constant_initializer(np.concatenate([np.eye(size), np.eye(size)]))
        output = tf.layers.dense(output, size, kernel_initializer=projection_initializer, name='projection')
    return output


def bi_lstm(size, sequence, seq_length, with_projection=False, **kwargs):
    fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(size)
    return _bi_rnn(size, fused_rnn, sequence, seq_length, with_projection)


def bi_rnn(size, rnn_cell, sequence, seq_length, with_projection=False, **kwargs):
    fused_rnn = tf.contrib.rnn.FusedRNNCellAdaptor(rnn_cell, use_dynamic_rnn=True)
    return _bi_rnn(size, fused_rnn, sequence, seq_length, with_projection)


def bi_sru(size, sequence, seq_length, with_residual=True, name='bi_sru', reuse=None, with_projection=False, **kwargs):
    """Simple Recurrent Unit, very fast.  https://openreview.net/pdf?id=rJBiunlAW."""
    fused_rnn = rnn.SRUFusedRNN(size, with_residual)
    return _bi_rnn(size, fused_rnn, sequence, seq_length, with_projection)


# Convolution Encoders


def convnet(repr_dim, inputs, num_layers, conv_width=3, activation=tf.nn.relu, **kwargs):
    # dim reduction
    output = inputs
    for i in range(num_layers):
        output = _convolutional_block(output, repr_dim, conv_width=conv_width, name="conv_%d" % i)
    return output


def _convolutional_block(inputs, out_channels, conv_width=3, name='conv', activation=tf.nn.relu, **kwargs):
    channels = inputs.get_shape()[2].value
    # [filter_height, filter_conv_width, in_channels, out_channels]
    f = tf.get_variable(name + '_filter', [conv_width, channels, out_channels])
    output = tf.nn.conv1d(inputs, f, 1, padding='SAME', name=name)
    return activation(output)


def depthwise_separable_convolution(repr_dim, inputs, conv_width, activation=tf.nn.relu, bias=True, **kwargs):
    inputs = tf.expand_dims(inputs, 1)
    shapes = inputs.shape.as_list()

    depthwise_filter = tf.get_variable("depthwise_filter",
                                       (1, conv_width, shapes[-1], 1),
                                       dtype=tf.float32)
    pointwise_filter = tf.get_variable("pointwise_filter", (1, 1, shapes[-1], repr_dim),
                                       dtype=tf.float32)
    outputs = tf.nn.separable_conv2d(inputs,
                                     depthwise_filter,
                                     pointwise_filter,
                                     strides=(1, 1, 1, 1),
                                     padding="SAME")
    outputs = tf.squeeze(outputs, 1)
    if bias:
        b = tf.get_variable("bias", outputs.shape[-1], initializer=tf.zeros_initializer())
        outputs += b
    outputs = activation(outputs)
    return outputs


# following implementation of fast encoding in https://openreview.net/pdf?id=HJRV1ZZAW
def _residual_dilated_convolution_block(inputs, dilation=1, conv_width=3, name="dilated_conv"):
    # [filter_height, filter_conv_width, in_channels, out_channels]
    output = inputs
    channels = inputs.get_shape()[2].value
    for i in range(2):
        # [filter_height, filter_conv_width, in_channels, out_channels]
        output = _convolutional_glu_block(output, channels, dilation, conv_width, name=name + '_' + str(i))
    return output + inputs


def _convolutional_glu_block(inputs, out_channels, dilation=1, conv_width=3, name='conv_glu', **kwargs):
    channels = inputs.get_shape()[2].value
    # [filter_height, filter_conv_width, in_channels, out_channels]
    f = tf.get_variable(name + '_filter', [1, conv_width, channels, out_channels * 2])
    output = tf.nn.atrous_conv2d(tf.expand_dims(inputs, 1), f, dilation, 'SAME', name=name)
    output = tf.squeeze(output, 1)
    output, gate = tf.split(output, 2, 2)
    output *= tf.sigmoid(gate)
    return output


def gated_linear_dilated_residual_network(out_size, inputs, dilations, conv_width=3, name='gldr_network', reuse=None,
                                          **kwargs):
    """Follows https://openreview.net/pdf?id=HJRV1ZZAW.

    Args:
        out_size: size of output
        inputs: input sequence tensor [batch_size, length, size]
        dilations: list, representing (half of the) network depth; each residual dilation block comprises 2 convolutions

    Returns:
        [batch_size, length, out_size] tensor
    """
    # dim reduction
    output = _convolutional_glu_block(inputs, out_size, name='conv_dim_reduction')
    for i, d in enumerate(dilations):
        output = _residual_dilated_convolution_block(output, d, conv_width, name='dilated_conv_%d' % i)
    return output


def gated_linear_convnet(out_size, inputs, num_layers, conv_width=3, **kwargs):
    """Follows https://openreview.net/pdf?id=HJRV1ZZAW.

    Args:
        out_size: size of output
        inputs: input sequence tensor [batch_size, length, size]
        num_layers: number of conv layers with conv_width

    Returns:
        [batch_size, length, out_size] tensor
    """
    # dim reduction
    output = inputs
    for i in range(num_layers):
        output = _convolutional_glu_block(output, out_size, conv_width=conv_width, name="conv_%d" % i)
    return output


def positional_encoding(inputs, lengths, **kwargs):
    repr_dim = inputs.get_shape()[-1].value
    pos = tf.reshape(tf.range(0.0, tf.to_float(tf.reduce_max(lengths)), dtype=tf.float32), [-1, 1])
    i = np.arange(0, repr_dim, 2, np.float32)
    denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
    enc = tf.expand_dims(tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
    return inputs + tf.tile(enc, [tf.shape(inputs)[0], 1, 1])


# Self attention layers
def self_attention(inputs, lengths, attn_type='bilinear', scaled=True, activation=None, with_sentinel=False, **kwargs):
    if attn_type == 'bilinear':
        attn_states = attention.bilinear_attention(
            inputs, inputs, lengths, scaled, with_sentinel, **kwargs)[2]
    elif attn_type == 'dot':
        attn_states = attention.dot_attention(
            inputs, inputs, lengths, scaled, with_sentinel, **kwargs)[2]
    elif attn_type == 'diagonal_bilinear':
        attn_states = \
            attention.diagonal_bilinear_attention(
                inputs, inputs, lengths, scaled, with_sentinel, **kwargs)[2]
    elif attn_type == 'mlp':
        attn_states = \
            attention.mlp_attention(
                kwargs['repr_dim'], activation, inputs, inputs, lengths, with_sentinel, **kwargs)[2]
    else:
        raise ValueError("Unknown attention type: %s" % attn_type)

    return attn_states
