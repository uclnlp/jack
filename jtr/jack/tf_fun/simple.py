import tensorflow as tf

def fully_connected_projection(inputs, output_size, name='projection_layer'):
    """
    Creates fully connected projection layer.
    Args:
        inputs (tensor): Input into the projection layer.
        output_size (int): Size of the targets (used in projection layer).
    """
    init = tf.contrib.layers.xavier_initializer(uniform=True) #uniform=False for truncated normal
    projection = tf.contrib.layers.fully_connected(inputs, output_size, weights_initializer=init, activation_fn=None)
    projection = tf.identity(projection, name=name)
    return projection
