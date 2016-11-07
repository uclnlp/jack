import tensorflow as tf


def random_truncated_exponential(w, num_samples):
    """
    Sample from truncated independent exponential distributions using weights w (or inverse exponential distribution for
    positive weight components).
    Args:
        w: a [num_rows, dim] matrix of `num_rows` weight vectors.
        num_samples: the number of samples to produce.

    Returns:
        x,p where `x` is a tensor [num_samples, num_rows, dim] of samples and `p` is a matrix [num_samples, num_rows]
        of the probabilities of each sample with respect to each row.
    """
    eps = 0.000001
    # w: [num_rows, dim]
    # returns: [num_samples, num_rows, dim] batch of samples drawn from a truncated exponential over [0,1] using w
    # as parameter
    shape = tf.concat(0, ((num_samples,), tf.shape(w)))  # [num_samples, num_rows, dim]
    f1 = tf.minimum(w, -eps)
    f2 = tf.maximum(w, eps)
    is_neg = tf.to_float(tf.less(w, 0.0))
    w_ = is_neg * f1 + (1.0 - is_neg) * f2
    u = tf.random_uniform(shape, 0.0, 1.0)  # [num_samples, num_rows, dim]
    z = (tf.exp(w_) - 1.0)  # [num_rows, dim]
    x = tf.log(z * u + 1) / w_
    #     p = tf.reduce_prod(tf.exp(x * w_) * w_ / z,0)
    # TODO: replace with robuster version
    #     p = tf.exp(tf.reduce_sum(tf.log(tf.exp(x * w_) * w_ / z),1))
    #     p = tf.exp(tf.reduce_sum(x * w_ + tf.log(w_) - tf.log(z),1))
    log_p_components = x * w_ + tf.log(tf.abs(w_)) - tf.log(tf.abs(z))
    p = tf.exp(tf.reduce_sum(log_p_components, 2))
    return x, p
