# -*- coding: utf-8 -*-

import tensorflow as tf

from jack.tfutil.segment import segment_softmax


def xqa_min_crossentropy_loss(start_scores, end_scores, answer_span, answer2support, support2question):
    """
    very common XQA loss function
    """
    num_questions = tf.reduce_max(support2question) + 1
    num_support = tf.unstack(tf.shape(start_scores))[0]

    start, end = answer_span[:, 0], answer_span[:, 1]

    start_probs = segment_softmax(start_scores, support2question)
    end_probs = segment_softmax(end_scores, support2question)

    start_probs = tf.gather_nd(start_probs, tf.stack([answer2support, start], 1))
    end_probs = tf.gather_nd(end_probs, tf.stack([answer2support, end], 1))

    span_probs = tf.unsorted_segment_sum(start_probs * end_probs, answer2support, num_support)
    span_probs = tf.unsorted_segment_sum(span_probs, support2question, num_questions)
    return -tf.reduce_mean(tf.log(span_probs + 1e-6)),


def xqa_min_crossentropy_span_loss(candidate_scores, span_candidates, answer_span, answer_to_question):
    """
    very common XQA loss function when predicting for entire spans
    """
    # align total spans and scores with correct answer spans
    span_candidates = tf.gather(span_candidates, answer_to_question)
    candidate_scores = tf.gather(candidate_scores, answer_to_question)

    # tile correct answers to num spans to find matching span
    answer_span_tiled = tf.expand_dims(answer_span, 1)

    span_labels = tf.cast(tf.reduce_all(tf.equal(answer_span_tiled, span_candidates), 2), tf.float32)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=candidate_scores, labels=span_labels)
    loss = tf.segment_min(loss, answer_to_question)
    return tf.reduce_mean(loss),
