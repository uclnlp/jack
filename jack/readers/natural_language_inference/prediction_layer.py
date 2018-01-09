import tensorflow as tf


def prediction_layer(hypothesis, hypothesis_length, premise, premise_length, num_classes, is_eval,
                     repr_dim=100, activation=None, module='max_avg_mlp', dropout=0.0, **kwargs):
    if module == 'max_avg_mlp':
        return max_avg_pool_prediction_layer(
            repr_dim, num_classes, hypothesis, hypothesis_length, premise, premise_length, is_eval, activation, dropout)
    elif module == 'max_mlp':
        return max_pool_prediction_layer(
            repr_dim, num_classes, hypothesis, hypothesis_length, premise, premise_length, is_eval, activation, dropout)
    elif module == 'max_mlp_hypothesis':
        return max_pool_prediction_layer_hypothesis(
            repr_dim, num_classes, hypothesis, hypothesis_length, is_eval, activation, dropout)
    elif module == 'max_interaction_mlp':
        return max_pool_interaction_prediction_layer(
            repr_dim, num_classes, hypothesis, hypothesis_length, premise, premise_length, is_eval, activation, dropout)
    else:
        raise ValueError("Unknown answer layer type: %s" % module)


def _mask(hypothesis, hypothesis_length, premise, premise_length):
    p_mask = tf.sequence_mask(premise_length, tf.shape(premise)[1], dtype=tf.float32)
    h_mask = tf.sequence_mask(hypothesis_length, tf.shape(hypothesis)[1], dtype=tf.float32)

    premise *= tf.expand_dims(p_mask, 2)
    hypothesis *= tf.expand_dims(h_mask, 2)
    return hypothesis, premise


def max_avg_pool_prediction_layer(size, num_classes, hypothesis, hypothesis_length, premise, premise_length, is_eval,
                                  activation=tf.tanh, dropout=0.0):
    hypothesis, premise = _mask(hypothesis, hypothesis_length, premise, premise_length)

    p_max = tf.reduce_max(premise, axis=1)
    p_avg = tf.reduce_sum(premise, axis=1) / tf.expand_dims(tf.to_float(premise_length), 1)

    h_max = tf.reduce_max(hypothesis, axis=1)
    h_avg = tf.reduce_sum(hypothesis, axis=1) / tf.expand_dims(tf.to_float(hypothesis_length), 1)

    inputs = tf.concat([p_max, p_avg, h_max, h_avg], 1)
    if dropout:
        inputs = tf.cond(is_eval, lambda: inputs, lambda: tf.nn.dropout(inputs, 1.0 - dropout))
    hidden = tf.layers.dense(inputs, size, activation)
    if dropout:
        hidden = tf.cond(is_eval, lambda: hidden, lambda: tf.nn.dropout(hidden, 1.0 - dropout))
    logits = tf.layers.dense(hidden, num_classes)

    return logits


def max_pool_prediction_layer(size, num_classes, hypothesis, hypothesis_length, premise, premise_length, is_eval,
                              activation=tf.tanh, dropout=0.0):
    hypothesis, premise = _mask(hypothesis, hypothesis_length, premise, premise_length)

    p_max = tf.reduce_max(premise, axis=1)

    h_max = tf.reduce_max(hypothesis, axis=1)

    inputs = tf.concat([p_max, h_max], 1)
    if dropout:
        inputs = tf.cond(is_eval, lambda: inputs, lambda: tf.nn.dropout(inputs, 1.0 - dropout))
    hidden = tf.layers.dense(inputs, size, activation)
    if dropout:
        hidden = tf.cond(is_eval, lambda: hidden, lambda: tf.nn.dropout(hidden, 1.0 - dropout))
    logits = tf.layers.dense(hidden, num_classes)

    return logits


def max_pool_interaction_prediction_layer(size, num_classes, hypothesis, hypothesis_length, premise, premise_length,
                                          is_eval, activation=tf.tanh, dropout=0.0):
    hypothesis, premise = _mask(hypothesis, hypothesis_length, premise, premise_length)

    p_max = tf.reduce_max(premise, axis=1)
    h_max = tf.reduce_max(hypothesis, axis=1)

    inputs = tf.concat([p_max, h_max, p_max - h_max, p_max * h_max], 1)
    if dropout:
        inputs = tf.cond(is_eval, lambda: inputs, lambda: tf.nn.dropout(inputs, 1.0 - dropout))
    hidden = tf.layers.dense(inputs, size, activation)
    if dropout:
        hidden = tf.cond(is_eval, lambda: hidden, lambda: tf.nn.dropout(hidden, 1.0 - dropout))
    logits = tf.layers.dense(hidden, num_classes)

    return logits


def max_pool_prediction_layer_hypothesis(size, num_classes, hypothesis, hypothesis_length, is_eval,
                                         activation=tf.tanh, dropout=0.0):
    h_mask = tf.sequence_mask(hypothesis_length, tf.shape(hypothesis)[1], dtype=tf.float32)
    hypothesis *= tf.expand_dims(h_mask, 2)

    h_max = tf.reduce_max(hypothesis, axis=1)

    hidden = tf.layers.dense(h_max, size, activation)
    if dropout:
        hidden = tf.cond(is_eval, lambda: hidden, lambda: tf.nn.dropout(hidden, 1.0 - dropout))
    logits = tf.layers.dense(hidden, num_classes)

    return logits
