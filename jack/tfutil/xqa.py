# -*- coding: utf-8 -*-

import tensorflow as tf

from jack.tfutil.segment import segment_softmax


def xqa_min_crossentropy_loss(start_scores, end_scores, answer_span, answer2support, support2question):
    """Very common XQA loss function."""
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
