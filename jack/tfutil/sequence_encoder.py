import numpy as np
import tensorflow as tf

from jack.tfutil import rnn


def encoder(sequence, seq_length, repr_dim=100, layer_type='lstm', num_layers=1, conv_width=3,
            dilations=None, reuse=False, with_projection=False, residual=False, name='encoder'):
    if num_layers == 1:
        if layer_type == 'lstm':
            out = bi_lstm(repr_dim, sequence, seq_length, name, reuse, with_projection)
        elif layer_type == 'sru':
            with_residual = sequence.get_shape()[2].value == repr_dim
            out = bi_sru(repr_dim, sequence, seq_length, with_residual, name, reuse, with_projection)
        elif layer_type == 'gru':
            out = bi_rnn(repr_dim, tf.contrib.rnn.BlockGRUCell(repr_dim), sequence,
                         seq_length, name, reuse, with_projection)
        elif layer_type == 'gldr':
            out = gated_linear_dilated_residual_network(
                repr_dim, sequence, dilations, conv_width, name, reuse)
        elif layer_type == 'convolution':
            out = gated_linear_convnet(repr_dim, sequence, num_layers, conv_width, name, reuse)
        elif layer_type == 'projection':
            out = tf.layers.dense(sequence, repr_dim, name=name)
        else:
            raise ValueError("Unknown encoder type: %s" % type)
    else:
        out = encoder(sequence, seq_length, repr_dim, layer_type, num_layers - 1, conv_width,
                      dilations, reuse, with_projection, name)
        out = encoder(out, seq_length, repr_dim, layer_type, 1, conv_width,
                      dilations, reuse, with_projection, name + str(num_layers - 1))
    if residual:
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

# following implementation of fast encoding in https://openreview.net/pdf?id=HJRV1ZZAW
def _residual_dilated_convolution_block(inputs, dilation=1, width=3, name="dilated_conv"):
    # [filter_height, filter_width, in_channels, out_channels]
    output = inputs
    channels = inputs.get_shape()[2].value
    for i in range(2):
        # [filter_height, filter_width, in_channels, out_channels]
        output = _convolutional_block_glu_block(output, channels, dilation, width, name=name + '_' + str(i))
    return output + inputs


def _convolutional_block_glu_block(inputs, out_channels, dilation=1, width=3, name='dilated_conv'):
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
        output = _convolutional_block_glu_block(inputs, out_size, name='conv_dim_reduction')
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
            output = _convolutional_block_glu_block(output, out_size, width=width, name="conv_%d" % i)
    return output
