# Contributing

In the following, you can see a simple baseline model for JTR:

TODO: describe nvocab, **options

```python
def boe_nosupport_cands_reader_model(placeholders, nvocab, **options):
    """
    Bag of embedding reader with pairs of (question, support) and candidates
    """

    # Model
    # [batch_size, max_seq1_length]
    question = placeholders['question']

    # [batch_size, candidate_size]
    targets = placeholders['targets']

    # [batch_size, max_num_cands]
    candidates = placeholders['candidates']

    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)
        varscope.reuse_variables()
        candidates_embedded = nvocab(candidates)

    logger.info('TRAINABLE VARIABLES (only embeddings): {}'.format(get_total_trainable_variables()))
    question_encoding = tf.reduce_sum(question_embedded, 1)

    scores = logits = tf.reduce_sum(tf.expand_dims(question_encoding, 1) * candidates_embedded, 2)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')

    logger.info('TRAINABLE VARIABLES (embeddings + model): {}'.format(get_total_trainable_variables()))
    logger.info('ALL VARIABLES (embeddings + model): {}'.format(get_total_variables()))

    return logits, loss, predict
```
