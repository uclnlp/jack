import tensorflow as tf


def highway_layer(arg, scope=None):
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1].value
        trans_gate = tf.contrib.layers.fully_connected(arg, 2*d, activation_fn=None, weights_initializer=None,
                                                       scope='trans_gate')
        trans, gate = tf.split(trans_gate, 2, len(arg.get_shape())-1)
        trans, gate = tf.tanh(trans), tf.sigmoid(gate)
        gate = tf.contrib.layers.fully_connected(arg, d, activation_fn=tf.sigmoid, weights_initializer=None,
                                                 scope='gate')
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers,scope=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev,scope="layer_{}".format(layer_idx))
            prev = cur
    return cur
