import tensorflow as tf


def segment_softmax(scores, partition):
    """Given scores and a partition, converts scores to probs by performing
    softmax over all rows within a partition."""

    # Subtract max
    num_partitions = tf.reduce_max(partition) + 1
    if len(scores.get_shape()) == 2:
        max_per_partition = tf.unsorted_segment_max(tf.reduce_max(scores, axis=1), partition, num_partitions)
        scores -= tf.expand_dims(tf.gather(max_per_partition, partition), axis=1)
    else:
        max_per_partition = tf.unsorted_segment_max(scores, partition, num_partitions)
        scores -= tf.gather(max_per_partition, partition)

    # Compute probs
    scores_exp = tf.exp(scores)
    if len(scores.get_shape()) == 2:
        scores_exp_sum_per_partition = tf.unsorted_segment_sum(tf.reduce_sum(scores_exp, axis=1), partition,
                                                               num_partitions)
        probs = scores_exp / tf.expand_dims(tf.gather(scores_exp_sum_per_partition, partition), axis=1)
    else:
        scores_exp_sum_per_partition = tf.unsorted_segment_sum(scores_exp, partition, num_partitions)
        probs = scores_exp / tf.gather(scores_exp_sum_per_partition, partition)

    return probs


def segment_argmax(input, partition):
    """Computes row and col indices Tensors of the segment max in the 2D input."""

    with tf.name_scope("segment_argmax"):
        num_partitions = tf.reduce_max(partition) + 1
        is_max = segment_is_max(input, partition)

        # The current is_max could still contain multiple True entries per
        # partition. As long as they are in the same row, that is not a problem.
        # However, we do need to remove duplicate Trues in the same partition
        # in multiple rows.
        # For that, we'll multiply is_max with the row indices + 1 and perform
        # segment_is_max() again.

        rows = tf.shape(input)[0]
        cols = tf.shape(input)[1]
        row_indices = tf.tile(tf.expand_dims(tf.range(rows), 1), [1, cols])
        is_max = segment_is_max(tf.cast(is_max, tf.int32) * (row_indices + 1), partition)

        # Get selected rows and columns
        row_selected = tf.reduce_any(is_max, axis=1)
        row_indices = tf.squeeze(tf.where(row_selected))
        rows_selected = tf.reduce_sum(tf.cast(row_selected, tf.int64))

        # Assert rows_selected is correct & ensure row_indices is always 1D
        with tf.control_dependencies([tf.assert_equal(rows_selected, num_partitions)]):
            row_indices = tf.reshape(row_indices, [-1])

        selected_rows_is_max = tf.gather(is_max, row_indices)
        col_indices = tf.argmax(tf.cast(selected_rows_is_max, tf.int64), axis=1)

        # Pack indices
        return row_indices, col_indices


def segment_is_max(inputs, segment_ids):
    num_segments = tf.reduce_max(segment_ids) + 1
    if len(inputs.get_shape()) > 1:
        inputs_max = tf.reduce_max(inputs, reduction_indices=list(range(1, len(inputs.get_shape()))))
    else:
        inputs_max = inputs
    max_per_partition = tf.unsorted_segment_max(inputs_max, segment_ids, num_segments)
    return tf.equal(inputs, tf.gather(max_per_partition, segment_ids))


def segment_sample_select(probs, segment_ids):
    num_segments = tf.reduce_max(segment_ids) + 1
    sampled = tf.random_uniform([num_segments])

    def scan_fn(acc, x):
        p, i = x[0], x[1]
        prev_v = tf.gather(acc[0], i)
        new_probs = acc[0] + tf.one_hot(i, num_segments, p)
        select = tf.logical_and(tf.less(prev_v, 0.0), tf.greater_equal(prev_v + p, 0.0))
        return new_probs, select

    _, selection = tf.scan(scan_fn, (probs, segment_ids), initializer=(-sampled, False))

    return selection
