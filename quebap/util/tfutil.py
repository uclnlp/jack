"""
         __  _ __
  __  __/ /_(_) /
 / / / / __/ / /
/ /_/ / /_/ / /
\__,_/\__/_/_/ v0.2
Making useful stuff happen since 2016
"""

import tensorflow as tf


def get_by_index(tensor, index):
    """
    :param tensor: [dim1 x dim2 x dim3] tensor
    :param index: [dim1] tensor of indices for dim2
    :return: [dim1 x dim3] tensor
    """
    dim1, dim2, dim3 = tf.unpack(tf.shape(tensor))
    flat_index = tf.range(0, dim1) * dim2 + (index - 1)
    return tf.gather(tf.reshape(tensor, [-1, dim3]), flat_index)


def get_last(tensor):
    """
    :param tensor: [dim1 x dim2 x dim3] tensor
    :return: [dim1 x dim3] tensor
    """
    shape = tf.shape(tensor)  # [dim1, dim2, dim3]
    slice_size = shape * [1, 0, 1] + [0, 1, 0]  # [dim1, 1 , dim3]
    slice_begin = shape * [0, 1, 0] + [0, -1, 0]  # [1, dim2-1, 1]
    return tf.squeeze(tf.slice(tensor, slice_begin, slice_size), [1])


def mask_for_lengths(lengths, batch_size=None, max_length=None, mask_right=True,
                     value=-1000.0):
    """
    Creates a [batch_size x max_length] mask.
    :param lengths: int64 1-dim tensor of batch_size lengths
    :param batch_size: int32 0-dim tensor or python int
    :param max_length: int32 0-dim tensor or python int
    :param mask_right: if True, everything before "lengths" becomes zero and the
        rest "value", else vice versa
    :param value: value for the mask
    :return: [batch_size x max_length] mask of zeros and "value"s
    """
    if max_length is None:
        max_length = tf.cast(tf.reduce_max(lengths), tf.int32)
    if batch_size is None:
        batch_size = tf.shape(lengths)[0]
    # [batch_size x max_length]
    mask = tf.reshape(tf.tile(tf.range(0, max_length), [batch_size]), tf.pack([batch_size, -1]))
    if mask_right:
        mask = tf.greater_equal(tf.cast(mask, tf.int64), tf.expand_dims(lengths, 1))
    else:
        mask = tf.less(tf.cast(mask, tf.int64), tf.expand_dims(lengths, 1))
    mask = tf.cast(mask, tf.float32) * value
    return mask


def tfrun(tensor):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        return sess.run(tensor)


def tfrunprint(tensor, suffix="", prefix=""):
    if prefix == "":
        print(tfrun(tensor), suffix)
    else:
        print(prefix, tfrun(tensor), suffix)


def tfrunprintshape(tensor, suffix="", prefix=""):
    tfrunprint(tf.shape(tensor), suffix, prefix)


def tfprint(tensor, fun=None, prefix=""):
    if fun is None:
        fun = lambda x: x
    return tf.Print(tensor, [fun(tensor)], prefix)


def tfprints(tensors, fun=None, prefix=""):
    if fun is None:
        fun = lambda x: x
    prints = []
    for i in range(0, len(tensors)):
        prints.append(tf.Print(tensors[i], [fun(tensors[i])], prefix))
    return prints


def tfprintshapes(tensors, prefix=""):
    return tfprints(tensors, lambda x: tf.shape(x), prefix)


def tfprintshape(tensor, prefix=""):
    return tfprint(tensor, lambda x: tf.shape(x), prefix)


def gather_in_dim(params, indices, dim, name=None):
    """
    Gathers slices in a defined dimension. If dim == 0 this is doing the same
      thing as tf.gather.
    """
    if dim == 0:
        return tf.gather(params, indices, name)
    else:
        dims = [i for i in range(0, len(params.get_shape()))]
        to_dims = list(dims)
        to_dims[0] = dim
        to_dims[dim] = 0

        transposed = tf.transpose(params, to_dims)
        gathered = tf.gather(transposed, indices)
        reverted = tf.transpose(gathered, to_dims)

        return reverted


def unit_length(tensor):
    l2norm_sq = tf.reduce_sum(tensor * tensor, 1, keep_dims=True)
    l2norm = tf.rsqrt(l2norm_sq)
    return tensor * l2norm