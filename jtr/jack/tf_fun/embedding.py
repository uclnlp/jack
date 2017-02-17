import math
from collections import defaultdict

import numpy as np
import tensorflow as tf

from jtr.util import tfutil


def conv_char_embeddings(vocab, size, word_ids, conv_width=5,
                         emb_initializer=tf.random_normal_initializer(0.0, 0.1), scope=None):
    """
    :param vocab: filled Vocab instance
    :param size: size of embeddings
    :param word_ids: tf.Tensor[None, None] or list of tensors
    :param conv_width: int
    :return: char embedded word ids
    """
    if not isinstance(word_ids, list):
        word_ids = [word_ids]

    # create character vocab + word lengths + char ids per word
    pad_right = math.ceil(conv_width / 2) # "fixed PAD o right side"
    vocab_size = max(vocab.sym2id.values()) + 1
    max_l = max(len(w) for w in vocab.sym2id) + pad_right
    char_vocab = defaultdict(lambda: len(char_vocab))
    char_vocab["PAD"] = 0
    word_to_chars_arr = np.zeros((vocab_size, max_l), np.int16)
    word_lengths_arr = np.zeros([vocab_size], np.int8)
    for w, i in vocab.sym2id.items():
        for k, c in enumerate(w):
            j = char_vocab[c]
            word_to_chars_arr[i, k] = j
        word_lengths_arr[i] = len(w) + conv_width - 1

    with tf.variable_scope(scope or "char_embeddings") as vs:
        word_to_chars = tf.constant(word_to_chars_arr, name="word_to_chars")
        word_lengths = tf.constant(word_lengths_arr, name="word_lengths")

        char_embedding_matrix = \
            tf.get_variable("char_embedding_matrix", shape=(len(char_vocab), size),
                            initializer=emb_initializer, trainable=True)

        all_embedded = []
        for i, ids in enumerate(zip(word_ids)):
            if i > 0:
                vs.reuse_variables()

            unique_words, word_idx = tf.unique(tf.reshape(ids, [-1]))
            chars = tf.nn.embedding_lookup(word_to_chars, unique_words)
            wl = tf.nn.embedding_lookup(word_lengths, unique_words)
            wl = tf.cast(wl, tf.int32)
            max_word_length = tf.reduce_max(wl)
            chars = tf.slice(chars, [0, 0], tf.stack([-1, max_word_length]))

            embedded_chars = tf.nn.embedding_lookup(char_embedding_matrix, tf.cast(chars, tf.int32))

            with tf.variable_scope("conv"):
                # create filter like this to get fan-in and fan-out right for initializers depending on those
                filter = tf.get_variable("filter", [conv_width*size, size])
                filter_reshaped = tf.reshape(filter, [conv_width, size, size])
                conv_out = tf.nn.conv1d(embedded_chars, filter_reshaped, 1, "SAME")
                conv_mask = tf.expand_dims(tfutil.mask_for_lengths(wl - pad_right, max_length=max_word_length), 2)
                conv_out = conv_out + conv_mask

            unique_embedded_words = tf.reduce_max(conv_out, [1])

            embedded_words = tf.gather(unique_embedded_words, word_idx)
            embedded_words = tf.reshape(embedded_words, tf.stack([-1, tf.unstack(tf.shape(ids))[1], size]))
            all_embedded.append(embedded_words)

    return all_embedded


def conv_char_embedding_alt(char_vocab, size, unique_word_chars, unique_word_lengths, word_to_uniqs,
                            conv_width=5, emb_initializer=tf.random_normal_initializer(0.0, 0.1), scope=None):
    # "fixed PADDING on character level"
    pad = tf.zeros(tf.stack([tf.shape(unique_word_lengths)[0], math.floor(conv_width / 2)]), tf.int32)
    unique_word_chars = tf.concat([pad, unique_word_chars, pad], 1)

    if not isinstance(word_to_uniqs, list):
        word_to_uniqs = [word_to_uniqs]

    with tf.variable_scope(scope or "char_embeddings") as vs:
        char_embedding_matrix = \
            tf.get_variable("char_embedding_matrix", shape=(len(char_vocab), size),
                            initializer=emb_initializer, trainable=True)

        max_word_length = tf.reduce_max(unique_word_lengths)
        embedded_chars = tf.nn.embedding_lookup(char_embedding_matrix, tf.cast(unique_word_chars, tf.int32))

        with tf.variable_scope("conv"):
            # create filter like this to get fan-in and fan-out right for initializers depending on those
            filter = tf.get_variable("filter", [conv_width*size, size])
            filter_reshaped = tf.reshape(filter, [conv_width, size, size])
            # [B, T, S + pad_right]
            conv_out = tf.nn.conv1d(embedded_chars, filter_reshaped, 1, "VALID")
            conv_mask = tf.expand_dims(tfutil.mask_for_lengths(unique_word_lengths, max_length=max_word_length), 2)
            conv_out = conv_out + conv_mask

        unique_embedded_words = tf.reduce_max(conv_out, [1])

        all_embedded = []
        for word_idx in word_to_uniqs:
            flat_word_idx = tf.reshape(word_idx, [-1])
            embedded_words = tf.gather(unique_embedded_words, flat_word_idx)
            embedded_words = tf.reshape(embedded_words, tf.stack([-1, tf.unstack(tf.shape(word_idx))[1], size]))
            all_embedded.append(embedded_words)

    return all_embedded
