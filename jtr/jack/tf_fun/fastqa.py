import tensorflow as tf

from jtr.jack.tf_fun.dropout import fixed_dropout
from jtr.jack.tf_fun.embedding import conv_char_embeddings, conv_char_embedding_alt
from jtr.jack.tf_fun.highway import highway_network
from jtr.jack.tf_fun.rnn import birnn_with_projection
from jtr.util import tfutil


def fastqa_model(shared_resources, emb_question, question_length,
                 emb_support, support_length,
                 unique_word_chars, unique_word_char_length,
                 question_words2unique, support_words2unique,
                 word_in_question,
                 correct_start, answer2question, keep_prob, is_eval):
    """
    fast_qa model
    Args:
        shared_resources: has at least a field config (dict) with keys "rep_dim", "rep_input_dim"
        emb_question: [Q, L_q, N]
        question_length: [Q]
        emb_support: [Q, L_s, N]
        support_length: [Q]
        unique_word_chars
        unique_word_char_length
        question_words2unique
        support_words2unique
        word_in_question: [Q, L_s]
        correct_start: [A], only during training, i.e., is_eval=False
        answer2question: [A], only during training, i.e., is_eval=False
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

        input_size = shared_resources.config["repr_input_dim"]
        size = shared_resources.config["repr_dim"]

        # set shapes for inputs
        emb_question.set_shape([None, None, input_size])
        emb_support.set_shape([None, None, input_size])

        # compute combined embeddings
        [char_emb_question, char_emb_support] = conv_char_embedding_alt(shared_resources.config["char_vocab"], size,
                                                                        unique_word_chars, unique_word_char_length,
                                                                        [question_words2unique, support_words2unique])

        emb_question = tf.concat(2, [emb_question, char_emb_question])
        emb_support = tf.concat(2, [emb_support, char_emb_support])

        # compute encoder features
        question_features = tf.ones(tf.pack([batch_size, max_question_length, 2]))

        v_wiqw = tf.get_variable("v_wiq_w", [1, 1, input_size + size],
                                 initializer=tf.constant_initializer(1.0))

        wiq_w = tf.batch_matmul(emb_question * v_wiqw, emb_support, adj_y=True)
        wiq_w = wiq_w + tf.expand_dims(support_mask, 1)

        wiq_w = tf.reduce_sum(tf.nn.softmax(wiq_w) * tf.expand_dims(question_binary_mask, 2), [1])

        # [B, L , 1]
        support_features = tf.concat(2, [tf.expand_dims(word_in_question, 2), tf.expand_dims(wiq_w,  2)])

        # highway layer to allow for interaction between concatenated embeddings
        all_embedded_hw = highway_network(tf.concat(1, [emb_question, emb_support]), 1)

        emb_question_hw = tf.slice(all_embedded_hw, [0, 0, 0], tf.pack([-1, max_question_length, -1]))
        emb_support_hw = tf.slice(all_embedded_hw, tf.pack([0, max_question_length, 0]), [-1, -1, -1])

        emb_question_hw.set_shape([None, None, input_size + size])
        emb_support_hw.set_shape([None, None, input_size + size])

        # variational dropout
        dropout_shape = tf.unpack(tf.shape(emb_question_hw))
        dropout_shape[1] = 1

        [emb_question_hw, emb_support_hw] = tf.cond(is_eval,
                                                    lambda: [emb_question_hw, emb_support_hw],
                                                    lambda: fixed_dropout([emb_question, emb_support_hw],
                                                                          keep_prob, dropout_shape))

        # extend embeddings with features
        emb_question_ext = tf.concat(2, [emb_question_hw, question_features])
        emb_support_ext = tf.concat(2, [emb_support_hw, support_features])

        # encode question and support
        rnn = tf.contrib.rnn.LSTMBlockFusedCell
        encoded_question = birnn_with_projection(size, rnn,
                                                 emb_question_ext, question_length,
                                                 projection_scope="question_proj")

        encoded_support = birnn_with_projection(size, rnn,
                                                emb_support_ext, support_length,
                                                share_rnn=True, projection_scope="support_proj")

        start_scores, end_scores, predicted_start_pointer, predicted_end_pointer = \
            fastqa_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                                correct_start, answer2question, is_eval)

        span = tf.concat(1, [tf.expand_dims(predicted_start_pointer, 1),
                             tf.expand_dims(predicted_end_pointer, 1)])

        return start_scores, end_scores, span


# ANSWER LAYER
def fastqa_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                        correct_start, answer2question, is_eval):
    batch_size = tf.shape(question_length)[0]
    input_size = encoded_support.get_shape()[-1].value
    support_states_flat = tf.reshape(encoded_support, [-1, input_size])
    offsets = tf.cast(tf.range(0, batch_size) * (tf.reduce_max(support_length)), dtype=tf.int64)

    # computing single time attention over question
    attention_scores = tf.contrib.layers.fully_connected(encoded_question, 1,
                                                         activation_fn=None,
                                                         weights_initializer=None,
                                                         biases_initializer=None,
                                                         scope="question_attention")
    q_mask = tfutil.mask_for_lengths(question_length, batch_size)
    attention_scores = attention_scores + tf.expand_dims(q_mask, 2)
    question_attention_weights = tf.nn.softmax(attention_scores, 1, name="question_attention_weights")
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

    # use correct start during training, because p(end|start) should be optimized
    start_pointer = tf.cond(is_eval, lambda: predicted_start_pointer, lambda: correct_start)

    # gather states for training, where spans should be predicted using multiple correct start per answer
    def align_tensor_with_answers_per_question(t):
        return tf.cond(is_eval, lambda: t, lambda: tf.gather(t, answer2question))

    offsets = align_tensor_with_answers_per_question(offsets)
    u_s = tf.gather(support_states_flat, start_pointer + offsets)

    start_scores = align_tensor_with_answers_per_question(start_scores)
    start_input = align_tensor_with_answers_per_question(start_input)
    encoded_support = align_tensor_with_answers_per_question(encoded_support)
    question_state = align_tensor_with_answers_per_question(question_state)
    support_mask = align_tensor_with_answers_per_question(support_mask)

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
    end_scores = tf.cond(is_eval,
                         lambda: end_scores + tfutil.mask_for_lengths(tf.cast(predicted_start_pointer, tf.int32),
                                                                      batch_size,
                                                                      tf.reduce_max(support_length), mask_right=False),
                         lambda: end_scores)

    predicted_end_pointer = tf.argmax(end_scores, 1)

    return start_scores, end_scores, predicted_start_pointer, predicted_end_pointer