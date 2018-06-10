# -*- coding: utf-8 -*-

import tensorflow as tf

from jack.util.tf import misc


def conv_char_embedding(num_chars, repr_dim, word_chars, word_lengths, word_sequences=None,
                        conv_width=5, emb_initializer=tf.random_normal_initializer(0.0, 0.1), scope=None):
    """Build simple convolutional character based embeddings for words with a fixed filter and size.

    After the convolution max-pooling over characters is employed for each filter. If word sequences are given,
    these will be embedded with the newly created embeddings.
    """
    # "fixed PADDING on character level"
    pad = tf.zeros(tf.stack([tf.shape(word_lengths)[0], conv_width // 2]), tf.int32)
    word_chars = tf.concat([pad, word_chars, pad], 1)

    with tf.variable_scope(scope or "char_embeddings"):
        char_embedding_matrix = \
            tf.get_variable("char_embedding_matrix", shape=(num_chars, repr_dim),
                            initializer=emb_initializer, trainable=True)

        max_word_length = tf.reduce_max(word_lengths)
        embedded_chars = tf.nn.embedding_lookup(char_embedding_matrix, tf.cast(word_chars, tf.int32))

        with tf.variable_scope("conv"):
            # create filter like this to get fan-in and fan-out right for initializers depending on those
            filter = tf.get_variable("filter", [conv_width * repr_dim, repr_dim])
            filter_reshaped = tf.reshape(filter, [conv_width, repr_dim, repr_dim])
            # [B, T, S + pad_right]
            conv_out = tf.nn.conv1d(embedded_chars, filter_reshaped, 1, "VALID")
            conv_mask = tf.expand_dims(misc.mask_for_lengths(word_lengths, max_length=max_word_length), 2)
            conv_out = conv_out + conv_mask

        embedded_words = tf.reduce_max(conv_out, 1)

    if word_sequences is None:
        return embedded_words

    if not isinstance(word_sequences, list):
        word_sequences = [word_sequences]
    all_embedded = []
    for word_idxs in word_sequences:
        all_embedded.append(tf.nn.embedding_lookup(embedded_words, word_idxs))

    return all_embedded


def conv_char_embedding_multi_filter(
        num_chars, filter_sizes, embedding_size, word_chars, word_lengths, word_sequences=None,
        emb_initializer=tf.random_normal_initializer(0.0, 0.1), projection_size=None, scope=None):
    """Build convolutional character based embeddings for words with multiple filters.

    Filter sizes is a list and each the position of each size in the list entry refers to its corresponding conv width.
    It can also be 0 (i.e., no filter of that conv width). E.g., sizes [4, 0, 7, 8] will create 4 conv filters of width
    1, no filter of width 2, 7 of width 3 and 8 of width 4. After the convolution max-pooling over characters is
    employed for each filter.

    embedding_size refers to the size of the character embeddings and projection size, if given, to the final size of
    the embedded characters after a final projection. If it is None, no projection will be applied and the resulting
    size is the sum of all filters.

    If word sequences are given, these will be embedded with the newly created embeddings.
    """
    with tf.variable_scope(scope or "char_embeddings"):
        char_embedding_matrix = \
            tf.get_variable("char_embedding_matrix", shape=(num_chars, embedding_size),
                            initializer=emb_initializer, trainable=True)

        pad = tf.zeros(tf.stack([tf.shape(word_lengths)[0], len(filter_sizes) // 2]), tf.int32)
        word_chars = tf.concat([pad, word_chars, pad], 1)

        max_word_length = tf.reduce_max(word_lengths)
        embedded_chars = tf.nn.embedding_lookup(char_embedding_matrix, tf.cast(word_chars, tf.int32))
        conv_mask = tf.expand_dims(misc.mask_for_lengths(word_lengths, max_length=max_word_length), 2)

        embedded_words = []
        for i, size in enumerate(filter_sizes):
            if size == 0:
                continue
            conv_width = i + 1
            with tf.variable_scope("conv_%d" % conv_width):
                # create filter like this to get fan-in and fan-out right for initializers depending on those
                filter = tf.get_variable("filter", [conv_width * embedding_size, size])
                filter_reshaped = tf.reshape(filter, [conv_width, embedding_size, size])
                cut = len(filter_sizes) // 2 - conv_width // 2
                embedded_chars_conv = embedded_chars[:, cut:-cut, :] if cut else embedded_chars
                conv_out = tf.nn.conv1d(embedded_chars_conv, filter_reshaped, 1, "VALID")
                conv_out += conv_mask
                embedded_words.append(tf.reduce_max(conv_out, 1))

        embedded_words = tf.concat(embedded_words, 1)
        if projection_size is not None:
            embedded_words = tf.layers.dense(embedded_words, projection_size)

    if word_sequences is None:
        return embedded_words

    if not isinstance(word_sequences, list):
        word_sequences = [word_sequences]
    all_embedded = []
    for word_idxs in word_sequences:
        embedded_words = tf.nn.embedding_lookup(embedded_words, word_idxs)
        all_embedded.append(embedded_words)

    return all_embedded
