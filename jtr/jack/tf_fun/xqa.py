import tensorflow as tf


def xqa_min_crossentropy_loss(start_scores, end_scores, answer_span, answer_to_question):
    """
    very common XQA loss function
    """
    start, end = [tf.squeeze(t, 1) for t in tf.split(1, 2, answer_span)]
    start_scores = tf.gather(start_scores, answer_to_question)
    end_scores = tf.gather(end_scores, answer_to_question)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(start_scores, start) + \
           tf.nn.sparse_softmax_cross_entropy_with_logits(end_scores, end)
    loss = tf.segment_min(loss, answer_to_question)
    return [tf.reduce_mean(loss)]