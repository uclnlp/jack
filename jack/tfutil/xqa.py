# -*- coding: utf-8 -*-

import tensorflow as tf

from jack.tfutil.segment import segment_softmax


def xqa_min_crossentropy_loss(start_scores, end_scores, answer_span, answer2support, support2question):
    """Very common XQA loss function."""
    num_questions = tf.reduce_max(support2question) + 1

    start, end = answer_span[:, 0], answer_span[:, 1]

    start_probs = segment_softmax(start_scores, support2question)
    start_probs = tf.gather_nd(start_probs, tf.stack([answer2support, start], 1))

    # only start probs are normalized on multi-paragraph, end probs conditioned on start only on per support level
    num_answers = tf.shape(answer_span)[0]
    is_aligned = tf.equal(tf.shape(end_scores)[0], num_answers)
    end_logprobs = tf.cond(
        is_aligned,
        lambda: tf.gather_nd(tf.nn.log_softmax(end_scores), tf.stack([tf.range(num_answers, dtype=tf.int32), end], 1)),
        lambda: tf.log(
            tf.gather_nd(segment_softmax(end_scores, support2question), tf.stack([answer2support, end], 1)) + 1e-6)
    )

    answer2question = tf.gather(support2question, answer2support)
    span_logprobs = tf.unsorted_segment_max(
        tf.log(start_probs + 1e-6) + end_logprobs, answer2question, num_questions)
    # infinite if there is no gold standard answer
    is_finite = tf.is_finite(span_logprobs)
    span_logprobs = tf.where(is_finite, span_logprobs, tf.zeros_like(span_logprobs))
    return -tf.reduce_mean(span_logprobs),
