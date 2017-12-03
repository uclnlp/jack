import tensorflow as tf

from jack.tfutil.misc import mask_for_lengths


def interaction_layer(support, support_length, question, question_length, support2question,
                      layer_type='attention_matching'):
    if layer_type == 'attention_matching':
        return attention_matching_layer(support, support_length, question, question_length, support2question)
    elif layer_type == 'bidaf':
        return bidaf_layer(support, support_length, question, question_length, support2question)
    else:
        raise ValueError("Unknown interaction type: %s" % layer_type)


def bidaf_layer(support, support_length, question, question_length, support2question):
    """Encodes support conditioned on question, e.g., using word-by-word attention."""
    question = tf.gather(question, support2question)
    size = support.get_shape()[-1].value

    # a_i,j= w1 * s_i + w2 * q_j + w3 * q_j * s_i
    w_3 = tf.get_variable("w3", [size, 1])
    question_mul = question * tf.reshape(w_3, [1, 1, size])

    # a = [batch, support_length, question_length]
    a = (tf.einsum('ijk,ilk->ijl', support, question_mul) +
         tf.layers.dense(support, 1) + tf.reshape(tf.layers.dense(question, 1), [-1, 1, tf.shape(question)[1]]))

    a += tf.expand_dims(mask_for_lengths(support_length), 2)
    a += tf.expand_dims(mask_for_lengths(question_length), 1)

    # support to question attention
    att_question = tf.nn.softmax(a, 2)  # softmax over question for each support token
    question_weighted = tf.einsum('ijl,ilk->ijk', att_question, question)

    # support to question attention
    # 1. filter important context words with max
    # 2. softmax over question to get the question words which are most relevant for the most relevant context words
    # max(a) = [batch, length1, length2] -> [ batch, length1] = most important context
    max_support = tf.reduce_max(a, 2)
    support_attention = tf.nn.softmax(max_support, 1)
    support_weighted = tf.einsum('ij,ijk->ik', support_attention, support)
    support_weighted = tf.expand_dims(support_weighted, 1)
    support_weighted = tf.tile(support_weighted, [1, tf.shape(support)[1], 1])

    return tf.concat([support, question_weighted, support * question_weighted, support * support_weighted], 2)


def attention_matching_layer(size, support, support_length, question, question_length, support2question):
    """Encodes support conditioned on question, e.g., using word-by-word attention."""
    question = tf.gather(question, support2question)

    # a_i,j= w1 * s_i + w2 * q_j + w3 * q_j * s_i
    w_3 = tf.get_variable("w3", [size, 1])
    question_mul = question * tf.reshape(w_3, [1, 1, size])

    # a = [batch, support_length, question_length]
    a = (tf.einsum('ijk,ilk->ijl', support, question_mul) +
         tf.layers.dense(support, 1) + tf.reshape(tf.layers.dense(question, 1), [-1, 1, tf.shape(question)[1]]))

    a += tf.expand_dims(mask_for_lengths(support_length), 2)
    a += tf.expand_dims(mask_for_lengths(question_length), 1)

    # support to question attention
    att_question = tf.nn.softmax(a, 2)  # softmax over question for each support token
    question_weighted = tf.einsum('ijl,ilk->ijk', att_question, question)

    return tf.concat([support, question_weighted], 2)
