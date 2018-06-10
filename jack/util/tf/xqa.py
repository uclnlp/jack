# -*- coding: utf-8 -*-

import tensorflow as tf

from jack.util.tf.segment import segment_softmax


def xqa_crossentropy_loss(start_scores, end_scores, answer_span, answer2support, support2question, use_sum=True):
    """Very common XQA loss function."""
    num_questions = tf.reduce_max(support2question) + 1

    start, end = answer_span[:, 0], answer_span[:, 1]

    start_probs = segment_softmax(start_scores, support2question)
    start_probs = tf.gather_nd(start_probs, tf.stack([answer2support, start], 1))

    # only start probs are normalized on multi-paragraph, end probs conditioned on start only on per support level
    num_answers = tf.shape(answer_span)[0]
    is_aligned = tf.equal(tf.shape(end_scores)[0], num_answers)
    end_probs = tf.cond(
        is_aligned,
        lambda: tf.gather_nd(tf.nn.softmax(end_scores), tf.stack([tf.range(num_answers, dtype=tf.int32), end], 1)),
        lambda: tf.gather_nd(segment_softmax(end_scores, support2question), tf.stack([answer2support, end], 1))
    )

    answer2question = tf.gather(support2question, answer2support)
    # compute losses individually
    if use_sum:
        span_probs = tf.unsorted_segment_sum(
            start_probs, answer2question, num_questions) * tf.unsorted_segment_sum(
            end_probs, answer2question, num_questions)
    else:
        span_probs = tf.unsorted_segment_max(
            start_probs, answer2question, num_questions) * tf.unsorted_segment_max(
            end_probs, answer2question, num_questions)

    return -tf.reduce_mean(tf.log(tf.maximum(1e-6, span_probs + 1e-6)))
