import tensorflow as tf

from jack.tfutil import attention
from jack.tfutil.misc import mask_for_lengths


def interaction_layer(support, support_length, question, question_length, support2question,
                      layer_type='attention_matching', attn_type='bilinear_diagonal', scaled=True, gated=True,
                      name='interaction_layer', reuse=False):
    with tf.variable_scope(name, reuse):
        question = tf.gather(question, support2question)
        question_length = tf.gather(question_length, support2question)
        if layer_type == 'attention_matching':
            return attention_matching_layer(support, support_length, question, question_length, support2question,
                                            attn_type, scaled, gated)
        elif layer_type == 'bidaf':
            return bidaf_layer(support, support_length, question, question_length, support2question)
        else:
            raise ValueError("Unknown interaction type: %s" % layer_type)


def bidaf_layer(support, support_length, question, question_length, support2question):
    """Encodes support conditioned on question, e.g., using word-by-word attention."""
    attn_scores, attn_probs, question_weighted = attention.diagonal_bilinear_attention(
        support, question, question_length, False)

    attn_scores += tf.expand_dims(mask_for_lengths(support_length, tf.shape(attn_scores)[1]), 2)

    max_support = tf.reduce_max(attn_scores, 2)
    support_attention = tf.nn.softmax(max_support, 1)
    support_weighted = tf.einsum('ij,ijk->ik', support_attention, support)
    support_weighted = tf.expand_dims(support_weighted, 1)
    support_weighted = tf.tile(support_weighted, [1, tf.shape(support)[1], 1])

    return tf.concat([support, question_weighted, support * question_weighted, support * support_weighted], 2)


def attention_matching_layer(support, support_length, question, question_length, support2question,
                             attn_type='diagonal_bilinear', scaled=True, gated=False):
    """Encodes support conditioned on question, e.g., using word-by-word attention."""
    if attn_type == 'bilinear':
        _, _, attn_states = attention.bilinear_attention(support, question, question_length, scaled)
    elif attn_type == 'dot':
        _, _, attn_states = attention.dot_attention(support, question, question_length, scaled)
    elif attn_type == 'diagonal_bilinear':
        _, _, attn_states = attention.diagonal_bilinear_attention(support, question, question_length, scaled)
    elif attn_type == 'mlp':
        _, _, attn_states = attention.mlp_attention(support.get_shape()[-1].value, tf.nn.relu, support, question,
                                                    question_length)
    else:
        raise ValueError("Unknown attention type: %s" % attn_type)

    if gated:
        g = tf.sigmoid(tf.einsum('abc,abc->ab',
                                 tf.layers.dense(support, attn_states.get_shape()[-1].value), attn_states))
        attn_states *= tf.expand_dims(g, 2)

    return tf.concat([support, attn_states], 2)
