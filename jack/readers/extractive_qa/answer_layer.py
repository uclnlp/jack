import tensorflow as tf

from jack.tfutil import misc
from jack.tfutil.segment import segment_top_k


def mlp_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                     correct_start, support2question, answer2support, is_eval, beam_size=1, max_span_size=16):
    """Answer layer for multiple paragraph QA."""
    max_support_length = tf.shape(encoded_support)[1]

    # computing single time attention over question
    attention_scores = tf.layers.dense(encoded_question, 1, name="question_attention")
    q_mask = misc.mask_for_lengths(question_length)
    attention_scores = attention_scores + tf.expand_dims(q_mask, 2)
    question_attention_weights = tf.nn.softmax(attention_scores, 1, name="question_attention_weights")
    question_state = tf.reduce_sum(question_attention_weights * encoded_question, [1])

    # Prediction
    # start
    static_input = tf.concat([tf.expand_dims(question_state, 1) * encoded_support,
                              encoded_support], 2)

    hidden = tf.layers.dense(question_state, 2 * size, name="hidden_1")
    hidden = tf.layers.dense(
        static_input, 2 * size, use_bias=False, name="hidden_2") + tf.expand_dims(hidden, 1)

    hidden_start, hidden_end = tf.split(hidden, 2, 2)

    support_mask = misc.mask_for_lengths(support_length)

    start_scores = tf.layers.dense(tf.nn.relu(hidden_start, name='hidden'), 1, use_bias=False, name="start_scores")
    start_scores = tf.squeeze(start_scores, [2])
    start_scores = start_scores + support_mask

    end_scores = tf.layers.dense(tf.nn.relu(hidden_end), 1, use_bias=False, name="end_scores")
    end_scores = tf.squeeze(end_scores, [2])
    end_scores = end_scores + support_mask

    _, _, num_doc_per_question = tf.unique_with_counts(support2question)
    offsets = tf.cumsum(num_doc_per_question, exclusive=True)
    doc_idx_for_support = tf.range(tf.shape(support2question)[0]) - tf.gather(offsets, support2question)

    def train():
        start_pointer = correct_start
        gathered_end_scores = tf.gather(end_scores, answer2support)
        gathered_start_scores = tf.gather(start_scores, answer2support)

        # assuming we know the correct start we only consider ends after that
        left_mask = misc.mask_for_lengths(tf.cast(start_pointer, tf.int32), max_support_length, mask_right=False)
        gathered_end_scores = gathered_end_scores + left_mask

        predicted_start_pointer = tf.argmax(gathered_start_scores, axis=1, output_type=tf.int32)
        predicted_end_pointer = tf.argmax(gathered_end_scores, axis=1, output_type=tf.int32)

        return (start_scores, end_scores,
                tf.gather(doc_idx_for_support, answer2support), predicted_start_pointer, predicted_end_pointer)

    def eval():
        # [num_questions, beam_size]
        doc_idx, beam_start_pointer, beam_start_scores = segment_top_k(start_scores, support2question, beam_size)

        # [num_questions * beam_size]
        doc_idx_flat = tf.reshape(doc_idx, [-1])
        beam_start_pointer_flat = tf.reshape(beam_start_pointer, [-1])

        # [num_questions * beam_size, support_length]
        beam_end_scores_flat = tf.gather(end_scores, doc_idx_flat)

        left_mask = misc.mask_for_lengths(tf.cast(beam_start_pointer_flat, tf.int32),
                                          max_support_length, mask_right=False)
        right_mask = misc.mask_for_lengths(tf.cast(beam_start_pointer_flat + max_span_size, tf.int32),
                                           max_support_length)
        beam_end_scores_flat = beam_end_scores_flat + left_mask + right_mask

        beam_end_pointer_flat = tf.argmax(beam_end_scores_flat, axis=1, output_type=tf.int32)

        return (start_scores, end_scores,
                tf.gather(doc_idx_for_support, doc_idx_flat), beam_start_pointer_flat, beam_end_pointer_flat)

    return tf.cond(is_eval, eval, train)
