import tensorflow as tf


def prediction_layer(hypothesis, hypothesis_length, premise, premise_length, num_classes, repr_dim=100, activation=None,
                     module='max_avg_mlp', **kwargs):
    if module == 'max_avg_mlp':
        return max_avg_pool_prediction_layer(
            repr_dim, num_classes, hypothesis, hypothesis_length, premise, premise_length, activation)
    elif module == 'max_mlp':
        return max_pool_prediction_layer(
            repr_dim, num_classes, hypothesis, hypothesis_length, premise, premise_length, activation)
    elif module == 'max_interaction_mlp':
        return max_pool_interaction_prediction_layer(
            repr_dim, num_classes, hypothesis, hypothesis_length, premise, premise_length, activation)
    else:
        raise ValueError("Unknown answer layer type: %s" % module)


def _mask(hypothesis, hypothesis_length, premise, premise_length):
    p_mask = tf.sequence_mask(premise_length, tf.shape(premise)[1], dtype=tf.float32)
    h_mask = tf.sequence_mask(hypothesis_length, tf.shape(hypothesis)[1], dtype=tf.float32)

    premise *= tf.expand_dims(p_mask, 2)
    hypothesis *= tf.expand_dims(h_mask, 2)
    return hypothesis, premise


def max_avg_pool_prediction_layer(size, num_classes, hypothesis, hypothesis_length, premise, premise_length,
                                  activation=tf.tanh):
    hypothesis, premise = _mask(hypothesis, hypothesis_length, premise, premise_length)

    p_max = tf.reduce_max(premise, axis=1)
    p_avg = tf.reduce_sum(premise, axis=1) / tf.expand_dims(tf.to_float(premise_length), 1)

    h_max = tf.reduce_max(hypothesis, axis=1)
    h_avg = tf.reduce_sum(hypothesis, axis=1) / tf.expand_dims(tf.to_float(hypothesis_length), 1)

    inputs = tf.concat([p_max, p_avg, h_max, h_avg], 1)
    hidden = tf.layers.dense(inputs, size, activation)
    logits = tf.layers.dense(hidden, num_classes)

    return logits


def max_pool_prediction_layer(size, num_classes, hypothesis, hypothesis_length, premise, premise_length,
                              activation=tf.tanh):
    hypothesis, premise = _mask(hypothesis, hypothesis_length, premise, premise_length)

    p_max = tf.reduce_max(premise, axis=1)

    h_max = tf.reduce_max(hypothesis, axis=1)

    inputs = tf.concat([p_max, h_max], 1)
    hidden = tf.layers.dense(inputs, size, activation)
    logits = tf.layers.dense(hidden, num_classes)

    return logits


def max_pool_interaction_prediction_layer(size, num_classes, hypothesis, hypothesis_length, premise, premise_length,
                                          activation=tf.tanh):
    hypothesis, premise = _mask(hypothesis, hypothesis_length, premise, premise_length)

    p_max = tf.reduce_max(premise, axis=1)
    h_max = tf.reduce_max(hypothesis, axis=1)

    inputs = tf.concat([p_max, h_max, p_max - h_max, p_max * h_max], 1)
    hidden = tf.layers.dense(inputs, size, activation)
    logits = tf.layers.dense(hidden, num_classes)

    return logits
