import tensorflow as tf

from jack.tfutil import misc
from jack.tfutil.segment import segment_top_k


def compute_question_state(encoded_question, question_length):
    attention_scores = tf.layers.dense(encoded_question, 1, name="question_attention")
    q_mask = misc.mask_for_lengths(question_length)
    attention_scores = attention_scores + tf.expand_dims(q_mask, 2)
    question_attention_weights = tf.nn.softmax(attention_scores, 1, name="question_attention_weights")
    question_state = tf.reduce_sum(question_attention_weights * encoded_question, 1)
    return question_state


def bilinear_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                          correct_start, support2question, answer2support, is_eval, beam_size=1,
                          max_span_size=10000):
    """Answer layer for multiple paragraph QA."""
    # computing single time attention over question
    question_state = compute_question_state(encoded_question, question_length)

    # compute logits
    hidden = tf.gather(tf.layers.dense(question_state, 2 * size, name="hidden"), support2question)
    hidden_start, hidden_end = tf.split(hidden, 2, 1)

    support_mask = misc.mask_for_lengths(support_length)

    start_scores = tf.einsum('ik,ijk->ij', hidden_start, encoded_support)
    start_scores = start_scores + support_mask

    end_scores = tf.einsum('ik,ijk->ij', hidden_end, encoded_support)
    end_scores = end_scores + support_mask

    return compute_spans(start_scores, end_scores, answer2support, is_eval, support2question,
                         correct_start, beam_size, max_span_size)


def mlp_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                     correct_start, support2question, answer2support, is_eval, beam_size=1, max_span_size=10000):
    """Answer layer for multiple paragraph QA."""
    # computing single time attention over question
    question_state = compute_question_state(encoded_question, question_length)

    # compute logits
    static_input = tf.concat([tf.gather(tf.expand_dims(question_state, 1), support2question) * encoded_support,
                              encoded_support], 2)

    hidden = tf.gather(tf.layers.dense(question_state, 2 * size, name="hidden_1"), support2question)
    hidden = tf.layers.dense(
        static_input, 2 * size, use_bias=False, name="hidden_2") + tf.expand_dims(hidden, 1)

    hidden_start, hidden_end = tf.split(hidden, 2, 2)

    support_mask = misc.mask_for_lengths(support_length)

    start_scores = tf.layers.dense(hidden_start, 1, use_bias=False, name="start_scores")
    start_scores = tf.squeeze(start_scores, [2])
    start_scores = start_scores + support_mask

    end_scores = tf.layers.dense(hidden_end, 1, use_bias=False, name="end_scores")
    end_scores = tf.squeeze(end_scores, [2])
    end_scores = end_scores + support_mask

    return compute_spans(start_scores, end_scores, answer2support, is_eval, support2question,
                         correct_start, beam_size, max_span_size)


def compute_spans(start_scores, end_scores, answer2support, is_eval, support2question,
                  correct_start=None, beam_size=1, max_span_size=10000):
    max_support_length = tf.shape(start_scores)[1]
    _, _, num_doc_per_question = tf.unique_with_counts(support2question)
    offsets = tf.cumsum(num_doc_per_question, exclusive=True)
    doc_idx_for_support = tf.range(tf.shape(support2question)[0]) - tf.gather(offsets, support2question)

    def train():
        gathered_end_scores = tf.gather(end_scores, answer2support)
        gathered_start_scores = tf.gather(start_scores, answer2support)

        if correct_start is not None:
            # assuming we know the correct start we only consider ends after that
            left_mask = misc.mask_for_lengths(tf.cast(correct_start, tf.int32), max_support_length, mask_right=False)
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


def conditional_answer_layer(size, encoded_question, question_length, encoded_support, support_length,
                             correct_start, support2question, answer2support, is_eval, beam_size=1, max_span_size=10000,
                             bilinear=False):
    question_state = compute_question_state(encoded_question, question_length)
    question_state = tf.gather(question_state, support2question)

    # Prediction
    # start
    if bilinear:
        hidden_start = tf.layers.dense(question_state, size, name="hidden_start")
        start_scores = tf.einsum('ik,ijk->ij', hidden_start, encoded_support)
    else:
        static_input = tf.concat([tf.expand_dims(question_state, 1) * encoded_support, encoded_support], 2)
        hidden_start = tf.layers.dense(question_state, size, name="hidden_start_1")
        hidden_start = tf.layers.dense(
            static_input, size, use_bias=False, name="hidden_start_2") + tf.expand_dims(hidden_start, 1)
        start_scores = tf.layers.dense(tf.nn.relu(hidden_start), 1, use_bias=False, name="start_scores")
        start_scores = tf.squeeze(start_scores, [2])

    support_mask = misc.mask_for_lengths(support_length)
    start_scores = start_scores + support_mask

    max_support_length = tf.shape(start_scores)[1]
    _, _, num_doc_per_question = tf.unique_with_counts(support2question)
    offsets = tf.cumsum(num_doc_per_question, exclusive=True)
    doc_idx_for_support = tf.range(tf.shape(support2question)[0]) - tf.gather(offsets, support2question)

    doc_idx, start_pointer = tf.cond(
        is_eval,
        lambda: segment_top_k(start_scores, support2question, beam_size)[:2],
        lambda: (tf.expand_dims(answer2support, 1), tf.expand_dims(correct_start, 1)))

    doc_idx_flat = tf.reshape(doc_idx, [-1])
    start_pointer = tf.reshape(start_pointer, [-1])

    start_state = tf.gather_nd(encoded_support, tf.stack([doc_idx_flat, start_pointer], 1))
    start_state.set_shape([None, size])

    encoded_support_gathered = tf.gather(encoded_support, doc_idx_flat)
    if bilinear:
        hidden_end = tf.layers.dense(tf.concat([question_state, start_state], 1), size, name="hidden_end")
        end_scores = tf.einsum('ik,ijk->ij', hidden_end, encoded_support_gathered)
    else:
        end_input = tf.concat([tf.expand_dims(start_state, 1) * encoded_support_gathered,
                               tf.gather(static_input, doc_idx_flat)], 2)

        hidden_end = tf.layers.dense(tf.concat([tf.gather(question_state, doc_idx_flat), start_state], 1), size,
                                     name="hidden_end_1")
        hidden_end = tf.layers.dense(
            end_input, size, use_bias=False, name="hidden_end_2") + tf.expand_dims(hidden_end, 1)

        end_scores = tf.layers.dense(tf.nn.relu(hidden_end), 1, use_bias=False, name="end_scores")
        end_scores = tf.squeeze(end_scores, [2])

    end_scores = end_scores + tf.gather(support_mask, doc_idx_flat)

    def train():
        predicted_end_pointer = tf.argmax(end_scores, axis=1, output_type=tf.int32)
        return start_scores, end_scores, doc_idx, start_pointer, predicted_end_pointer

    def eval():
        # [num_questions * beam_size, support_length]
        left_mask = misc.mask_for_lengths(tf.cast(start_pointer, tf.int32),
                                          max_support_length, mask_right=False)
        right_mask = misc.mask_for_lengths(tf.cast(start_pointer + max_span_size, tf.int32),
                                           max_support_length)
        masked_end_scores = end_scores + left_mask + right_mask
        predicted_ends = tf.argmax(masked_end_scores, axis=1, output_type=tf.int32)

        return (start_scores, masked_end_scores,
                tf.gather(doc_idx_for_support, doc_idx_flat), start_pointer, predicted_ends)

    return tf.cond(is_eval, eval, train)
