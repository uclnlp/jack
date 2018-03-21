import tensorflow as tf


def segment_softmax(scores, segment_ids):
    """Given scores and a partition, converts scores to probs by performing
    softmax over all rows within a partition."""

    # Subtract max
    num_segments = tf.reduce_max(segment_ids) + 1
    if len(scores.get_shape()) == 2:
        max_per_partition = tf.unsorted_segment_max(tf.reduce_max(scores, axis=1), segment_ids, num_segments)
        scores -= tf.expand_dims(tf.gather(max_per_partition, segment_ids), axis=1)
    else:
        max_per_partition = tf.unsorted_segment_max(scores, segment_ids, num_segments)
        scores -= tf.gather(max_per_partition, segment_ids)

    # Compute probs
    scores_exp = tf.exp(scores)
    if len(scores.get_shape()) == 2:
        scores_exp_sum_per_partition = tf.unsorted_segment_sum(tf.reduce_sum(scores_exp, axis=1), segment_ids,
                                                               num_segments)
        probs = scores_exp / tf.expand_dims(tf.gather(scores_exp_sum_per_partition, segment_ids), axis=1)
    else:
        scores_exp_sum_per_partition = tf.unsorted_segment_sum(scores_exp, segment_ids, num_segments)
        probs = scores_exp / tf.gather(scores_exp_sum_per_partition, segment_ids)

    return probs


def segment_argmax(input, segment_ids):
    """Computes row and col indices Tensors of the segment max in the 2D input."""

    with tf.name_scope("segment_argmax"):
        num_partitions = tf.reduce_max(segment_ids) + 1
        is_max = segment_is_max(input, segment_ids)

        # The current is_max could still contain multiple True entries per
        # partition. As long as they are in the same row, that is not a problem.
        # However, we do need to remove duplicate Trues in the same partition
        # in multiple rows.
        # For that, we'll multiply is_max with the row indices + 1 and perform
        # segment_is_max() again.

        rows = tf.shape(input)[0]
        cols = tf.shape(input)[1]
        row_indices = tf.tile(tf.expand_dims(tf.range(rows), 1), [1, cols])
        is_max = segment_is_max(tf.cast(is_max, tf.int32) * (row_indices + 1), segment_ids)

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


def segment_top_k(input, segment_ids, k):
    """Computes top k elements for segments in 2D input.

    segment_idx needs to be sorted.

    Returns:
        [num_segments, k]- tensors for rows, columns, scores of best k results in each segment
    """

    with tf.name_scope("segment_top_k"):
        all_top_k_scores, all_top_k_indices = tf.nn.top_k(input, k)
        rows = tf.tile(tf.expand_dims(tf.range(tf.shape(input)[0], dtype=tf.int32), 1), [1, k])

        result_rows = tf.zeros([k], tf.int32)
        result_columns = tf.zeros([k], tf.int32)
        result_scores = tf.zeros([k], tf.float32)

        def replace(old, new):
            return tf.concat([old[:-1], tf.expand_dims(new, 0)], 0)

        def scan_fn(acc, x):
            result_row, result_column, result_score, last_index = acc

            row_indices = x[0]
            segment_idx = x[1]
            top_k_scores = x[2]
            col_indices = x[3]

            def merge():
                new_result_row = tf.concat([result_row, row_indices], 0)
                new_result_column = tf.concat([result_column, col_indices], 0)
                new_result_score = tf.concat([result_score, top_k_scores], 0)
                new_result_score, new_top_k_indices = tf.nn.top_k(new_result_score, k)
                new_result_row = tf.gather(new_result_row, new_top_k_indices[:k])
                new_result_column = tf.gather(new_result_column, new_top_k_indices[:k])

                return new_result_row, new_result_column, new_result_score, segment_idx

            return tf.cond(segment_idx > last_index,
                           lambda: (row_indices, col_indices, top_k_scores, segment_idx),
                           merge)

        last_index = -1
        result_rows, result_columns, result_scores, _ = tf.scan(
            scan_fn, (rows, segment_ids, all_top_k_scores, all_top_k_indices),
            initializer=(result_rows, result_columns, result_scores, last_index))

        to_gather = tf.squeeze(tf.where((segment_ids[1:] - segment_ids[:-1]) > 0))
        to_gather = tf.concat([to_gather, tf.shape(segment_ids, out_type=tf.int64) - 1], 0)

        return tf.gather(result_rows, to_gather), tf.gather(result_columns, to_gather), tf.gather(result_scores,
                                                                                                  to_gather)
