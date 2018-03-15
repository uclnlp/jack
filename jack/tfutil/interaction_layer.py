import tensorflow as tf

from jack.tfutil import attention
from jack.tfutil import sequence_encoder
from jack.tfutil.misc import mask_for_lengths


def interaction_layer(seq1, seq1_length, seq2, seq2_length, seq1_to_seq2, seq2_to_seq1,
                      module='attention_matching', attn_type='bilinear_diagonal', scaled=True, with_sentinel=False,
                      name='interaction_layer', reuse=False, num_layers=1, encoder=None, concat=True,
                      repr_dim=None, **kwargs):
    with tf.variable_scope(name, reuse=reuse):
        if seq1_to_seq2 is not None:
            seq2 = tf.gather(seq2, seq1_to_seq2)
            seq2_length = tf.gather(seq2_length, seq1_to_seq2)
        if module == 'attention_matching':
            out = attention_matching_layer(seq1, seq1_length, seq2, seq2_length,
                                           attn_type, scaled, with_sentinel, seq2_to_seq1=seq2_to_seq1)
        elif module == 'bidaf':
            out = bidaf_layer(seq1, seq1_length, seq2, seq2_length, seq2_to_seq1=seq2_to_seq1)
        elif module == 'coattention':
            if 'repr_dim' not in encoder:
                encoder['repr_dim'] = repr_dim
            out = coattention_layer(
                seq1, seq1_length, seq2, seq2_length, attn_type, scaled, with_sentinel, seq2_to_seq1, num_layers,
                encoder)
        else:
            raise ValueError("Unknown interaction type: %s" % module)

    if concat:
        out = tf.concat([seq1, out], 2)
    return out


def bidaf_layer(seq1, seq1_length, seq2, seq2_length, seq2_to_seq1=None):
    """Encodes seq1 conditioned on seq2, e.g., using word-by-word attention."""
    attn_scores, attn_probs, seq2_weighted = attention.diagonal_bilinear_attention(
        seq1, seq2, seq2_length, False, seq2_to_seq1=seq2_to_seq1)

    attn_scores += tf.expand_dims(mask_for_lengths(seq1_length, tf.shape(attn_scores)[1]), 2)

    max_seq1 = tf.reduce_max(attn_scores, 2)
    seq1_attention = tf.nn.softmax(max_seq1, 1)
    seq1_weighted = tf.einsum('ij,ijk->ik', seq1_attention, seq1)
    seq1_weighted = tf.expand_dims(seq1_weighted, 1)
    seq1_weighted = tf.tile(seq1_weighted, [1, tf.shape(seq1)[1], 1])

    return tf.concat([seq2_weighted, seq1 * seq2_weighted, seq1 * seq1_weighted], 2)


def attention_matching_layer(seq1, seq1_length, seq2, seq2_length,
                             attn_type='diagonal_bilinear', scaled=True, with_sentinel=False, seq2_to_seq1=None):
    """Encodes seq1 conditioned on seq2, e.g., using word-by-word attention."""
    if attn_type == 'bilinear':
        _, _, attn_states = attention.bilinear_attention(seq1, seq2, seq2_length, scaled, with_sentinel,
                                                         seq2_to_seq1=seq2_to_seq1)
    elif attn_type == 'dot':
        _, _, attn_states = attention.dot_attention(seq1, seq2, seq2_length, scaled, with_sentinel,
                                                    seq2_to_seq1=seq2_to_seq1)
    elif attn_type == 'diagonal_bilinear':
        _, _, attn_states = attention.diagonal_bilinear_attention(seq1, seq2, seq2_length, scaled,
                                                                  with_sentinel, seq2_to_seq1=seq2_to_seq1)
    elif attn_type == 'mlp':
        _, _, attn_states = attention.mlp_attention(seq1.get_shape()[-1].value, tf.nn.relu, seq1, seq2,
                                                    seq2_length, with_sentinel, seq2_to_seq1=seq2_to_seq1)
    else:
        raise ValueError("Unknown attention type: %s" % attn_type)

    return attn_states


def coattention_layer(seq1, seq1_length, seq2, seq2_length,
                      attn_type='diagonal_bilinear', scaled=True, with_sentinel=False, seq2_to_seq1=None,
                      num_layers=1, encoder=None):
    """Encodes seq1 conditioned on seq2, e.g., using word-by-word attention."""
    if attn_type == 'bilinear':
        attn_fun = attention.bilinear_attention
    elif attn_type == 'dot':
        attn_fun = attention.dot_attention
    elif attn_type == 'diagonal_bilinear':
        attn_fun = attention.diagonal_bilinear_attention
    else:
        raise ValueError("Unknown attention type: %s" % attn_type)

    _, _, attn_states1, attn_states2, co_attn_state = attention.coattention(
        seq1, seq1_length, seq2, seq2_length, scaled, with_sentinel, attn_fun)

    if num_layers < 2:
        out = tf.concat([attn_states1, co_attn_state], 2)
    else:
        seq1, attn_states1, attn_states2, co_attn_state = [], [attn_states1], [attn_states2], [co_attn_state]
        for i in range(1, num_layers):
            with tf.variable_scope(str(i)):
                enc_1 = sequence_encoder.encoder(
                    attn_states1[-1], seq1_length, name='encoder1', **encoder)
                enc_2 = sequence_encoder.encoder(
                    attn_states2[-1], seq2_length, name='encoder2', **encoder)
                seq1.append(enc_1)
                _, _, new_attn_states1, new_attn_states2, new_co_attn_state = attention.coattention(
                    enc_1, seq1_length, enc_2, seq2_length, scaled, with_sentinel, attn_fun, seq2_to_seq1=seq2_to_seq1)
                attn_states1.append(new_attn_states1)
                attn_states2.append(new_attn_states2)
                co_attn_state.append(new_co_attn_state)
        out = tf.concat(seq1 + attn_states1 + co_attn_state, 2)
    # out.set_shape([None, None, (3 * num_layers - 1) * sum(s.get_shape()[-1].value for s in seq1) +
    #               seq2.get_shape()[-1].value])
    return out
