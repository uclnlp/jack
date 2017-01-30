import tensorflow as tf
import numpy as np
from jtr.util import tfutil


def fastqa_model(shared_resources, emb_question, question_length, emb_support, support_length, word_in_question,
                 keep_prob, is_eval):
    """
    fast_qa model
    Args:
        shared_resources: has at least a field config (dict) with key "rep_dim" -> int
        emb_question: [B, L_q, N]
        question_length: [B]
        emb_support: [B, L_s, N]
        support_length: [B]
        word_in_question: [B, L_s]
        keep_prob: []
        is_eval: []

    Returns:
        start_scores [B, L_s, N], end_scores [B, L_s, N], span_prediction [B, 2]
    """
    with tf.variable_scope("fast_qa", initializer=tf.contrib.layers.xavier_initializer()):
        # Some helpers
        batch_size = tf.shape(question_length)[0]
        max_question_length = tf.reduce_max(question_length)
        support_mask = tfutil.mask_for_lengths(support_length, batch_size)
        question_binary_mask = tfutil.mask_for_lengths(question_length, batch_size, mask_right=False, value=1.0)

        # compute encoder features
        question_features = tf.ones(tf.pack([batch_size, max_question_length, 2]))

        emb_size = emb_question.get_shape()[-1].value
        v_wiqw = tf.get_variable("v_wiq_w", [1, 1, emb_size], initializer=tf.constant_initializer(1.0))

        wiq_w = tf.batch_matmul(emb_question * v_wiqw, emb_support, adj_y=True)
        wiq_w = wiq_w + tf.expand_dims(support_mask, 1)

        wiq_w = tf.reduce_sum(tf.nn.softmax(wiq_w) * tf.expand_dims(question_binary_mask, 2), [1])

        # [B, L , 1]
        support_features = tf.concat(2, [tf.expand_dims(word_in_question, 2), tf.expand_dims(wiq_w,  2)])

        # variational dropout
        dropout_shape = tf.unpack(tf.shape(emb_question))
        dropout_shape[1] = 1

        emb_question, emb_support = tf.cond(is_eval,
                                            lambda: [emb_question, emb_support],
                                            lambda: fixed_dropout([emb_question, emb_support], keep_prob, dropout_shape))

        # extend embeddings with features
        emb_question_ext = tf.concat(2, [emb_question, question_features])
        emb_support_ext = tf.concat(2, [emb_support, support_features])

        # encode question and support
        rnn = tf.contrib.rnn.LSTMBlockFusedCell(shared_resources.config["repr_dim"])
        encoded_question = birnn_projection_layer(shared_resources.config["repr_dim"], rnn,
                                                  emb_question_ext, support_length,
                                                  projection_scope="question_proj")

        encoded_support = birnn_projection_layer(shared_resources.config["repr_dim"], rnn,
                                                 emb_support_ext, support_length,
                                                 share_rnn=True, projection_scope="context_proj")

        start_scores, end_scores, predicted_start_pointer, predicted_end_pointer, question_attention_weights = \
            fastqa_answer_layer(shared_resources.config["repr_dim"], encoded_question, question_length, encoded_support,
                                support_length)

        span = tf.concat(1, [tf.expand_dims(predicted_start_pointer, 1),
                             tf.expand_dims(predicted_end_pointer, 1)])

        return start_scores, end_scores, span


# PREPROCESSING

def fixed_dropout(xs, keep_prob, noise_shape, seed=None):
    """
    Apply dropout with same mask over all inputs
    Args:
        xs: list of tensors
        keep_prob:
        noise_shape:
        seed:

    Returns:
        list of dropped inputs
    """
    with tf.name_scope("dropout", values=xs):
        # Do nothing if we know keep_prob == 1
        if tf.tensor_util.constant_value(keep_prob) == 1:
          return xs

        noise_shape = noise_shape
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape, seed=seed, dtype=xs[0].dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)
        outputs = []
        for x in xs:
            ret = tf.div(x, keep_prob) * binary_tensor
            ret.set_shape(x.get_shape())
            outputs.append(ret)
        return outputs


def birnn_projection_layer(size, fused_rnn_constructor, inputs, length, share_rnn=False, projection_scope=None):
    projection_initializer = tf.constant_initializer(np.concatenate([np.eye(size), np.eye(size)]))
    fused_rnn = fused_rnn_constructor(size)
    with tf.variable_scope("RNN", reuse=share_rnn):
        encoded = fused_birnn(fused_rnn, inputs, sequence_length=length, dtype=tf.float32, time_major=False)[0]
        encoded = tf.concat(2, encoded)

    projected = tf.contrib.layers.fully_connected(encoded, size,
                                                  activation_fn=tf.tanh,
                                                  weights_initializer=projection_initializer,
                                                  scope=projection_scope)
    return projected


def fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state=None, dtype=None, scope=None, time_major=True):
    if not time_major:
        inputs = tf.transpose(inputs, [1, 0, 2])
    #assumes that time dim is 0 and batch is 1
    rev_inputs = tf.reverse_sequence(inputs, sequence_length, 0, 1)
    rev_outputs, last_state = fused_rnn(rev_inputs, sequence_length=sequence_length, initial_state=initial_state,
                                        dtype=dtype, scope=scope)
    outputs = tf.reverse_sequence(rev_outputs, sequence_length, 0, 1)
    if not time_major:
        outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs, last_state


def fused_birnn(fused_rnn, inputs, sequence_length, initial_state=None, dtype=None, scope=None, time_major=True,
                backward_device=None):
    with tf.variable_scope(scope or "BiRNN"):
        sequence_length = tf.cast(sequence_length, tf.int32)
        if not time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])
        outputs_fw, state_fw = fused_rnn(inputs, sequence_length=sequence_length, initial_state=initial_state,
                                         dtype=dtype, scope="FW")

        if backward_device is not None:
            with tf.device(backward_device):
                outputs_bw, state_bw = fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state, dtype, scope="BW")
        else:
            outputs_bw, state_bw = fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state, dtype, scope="BW")

        if not time_major:
            outputs_fw = tf.transpose(outputs_fw, [1, 0, 2])
            outputs_bw = tf.transpose(outputs_bw, [1, 0, 2])
    return (outputs_fw, outputs_bw), (state_fw, state_bw)


# ANSWER LAYER
def fastqa_answer_layer(size, encoded_question, question_length, encoded_support, support_length):
    batch_size = tf.shape(question_length)[0]
    input_size = encoded_support.get_shape()[-1].value
    support_states_flat = tf.reshape(encoded_support, [-1, input_size])
    offsets = tf.cast(tf.range(0, batch_size), dtype=tf.int64) * (tf.reduce_max(support_length))

    # computing single time attention over question
    attention_scores = tf.contrib.layers.fully_connected(encoded_question, 1,
                                                         activation_fn=None,
                                                         weights_initializer=None,
                                                         biases_initializer=None,
                                                         scope="question_attention")
    attention_scores = attention_scores + tf.expand_dims(tfutil.mask_for_lengths(question_length, batch_size), 2)
    question_attention_weights = tf.nn.softmax(attention_scores, 1)
    question_state = tf.reduce_sum(question_attention_weights * encoded_question, [1])

    # prediction

    #START
    start_input = tf.concat(2, [tf.expand_dims(question_state, 1) * encoded_support,
                                encoded_support])

    q_start_inter = tf.contrib.layers.fully_connected(question_state, size,
                                                      activation_fn=None,
                                                      weights_initializer=None,
                                                      scope="q_start_inter")

    q_start_state = tf.contrib.layers.fully_connected(start_input, size,
                                                      activation_fn=None,
                                                      weights_initializer=None,
                                                      scope="q_start") + tf.expand_dims(q_start_inter, 1)

    start_scores = tf.contrib.layers.fully_connected(tf.nn.relu(q_start_state), 1,
                                                     activation_fn=None,
                                                     weights_initializer=None,
                                                     biases_initializer=None,
                                                     scope="start_scores")
    start_scores = tf.squeeze(start_scores, [2])

    support_mask = tfutil.mask_for_lengths(support_length, batch_size)
    start_scores = start_scores + support_mask

    predicted_start_pointer = tf.argmax(start_scores, 1)

    start_pointer = predicted_start_pointer

    u_s = tf.gather(support_states_flat, start_pointer + offsets)

    #END
    end_input = tf.concat(2, [tf.expand_dims(u_s, 1) * encoded_support, start_input])

    q_end_inter = tf.contrib.layers.fully_connected(tf.concat(1, [question_state, u_s]), size,
                                                    activation_fn=None,
                                                    weights_initializer=None,
                                                    scope="q_end_inter")

    q_end_state = tf.contrib.layers.fully_connected(end_input, size,
                                                    activation_fn=None,
                                                    weights_initializer=None,
                                                    scope="q_end") + tf.expand_dims(q_end_inter, 1)

    end_scores = tf.contrib.layers.fully_connected(tf.nn.relu(q_end_state), 1,
                                                   activation_fn=None,
                                                   weights_initializer=None,
                                                   biases_initializer=None,
                                                   scope="end_scores")
    end_scores = tf.squeeze(end_scores, [2])
    end_scores = end_scores + support_mask

    predicted_end_pointer = tf.argmax(end_scores, 1)

    return start_scores, end_scores, predicted_start_pointer, predicted_end_pointer, question_attention_weights