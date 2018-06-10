import tensorflow as tf

from jack.util.tf import modular_encoder, misc
from jack.util.tf.embedding import conv_char_embedding


def embedding_refinement(size, word_embeddings, sequence_module, reading_sequence, reading_sequence_2_batch,
                         reading_sequence_lengths, word2lemma, unique_word_chars=None,
                         unique_word_char_length=None, is_eval=False, sequence_indices=None, num_sequences=4,
                         only_refine=False, keep_prob=1.0, batch_size=None, with_char_embeddings=False, num_chars=0):
    if batch_size is None:
        batch_size = tf.reduce_max(tf.stack([tf.shape(s)[0] if s2b is None else tf.reduce_max(s2b) + 1
                                             for s, s2b in zip(reading_sequence, reading_sequence_2_batch)]))

    sequence_indices = sequence_indices if sequence_indices is not None else list(range(len(reading_sequence)))

    if not only_refine:
        word_embeddings = tf.layers.dense(word_embeddings, size, activation=tf.nn.relu, name="embeddings_projection")
        if with_char_embeddings:
            word_embeddings = word_with_char_embed(
                size, word_embeddings, unique_word_chars, unique_word_char_length, num_chars)
        if keep_prob < 1.0:
            word_embeddings = tf.cond(is_eval,
                                      lambda: word_embeddings,
                                      lambda: tf.nn.dropout(word_embeddings, keep_prob, [1, size]))
        # tile word_embeddings by batch size (individual batches update embeddings individually)
        ctxt_word_embeddings = tf.tile(word_embeddings, tf.stack([batch_size, 1]))
        # HACK so that backprop works with indexed slices that come through here which are not handled by tile
        ctxt_word_embeddings *= 1.0
    else:
        ctxt_word_embeddings = word_embeddings

    num_words = tf.shape(word2lemma)[0]

    # divide uniq words for each question by offsets
    offsets = tf.expand_dims(tf.range(0, num_words * batch_size, num_words), 1)

    # each token is assigned a word idx + offset for distinguishing words between batch instances
    reading_sequence_offset = [
        s + offsets if s2b is None else s + tf.gather(offsets, s2b)
        for s, s2b in zip(reading_sequence, reading_sequence_2_batch)]

    word2lemma_off = tf.tile(tf.reshape(word2lemma, [1, -1]), [batch_size, 1]) + offsets
    word2lemma_off = tf.reshape(word2lemma_off, [-1])

    with tf.variable_scope("refinement") as vs:
        for i, seq, length in zip(sequence_indices, reading_sequence_offset, reading_sequence_lengths):
            if i > 0:
                vs.reuse_variables()
            num_seq = tf.shape(length)[0]

            def non_zero_batchsize_op():
                max_length = tf.shape(seq)[1]
                encoded = tf.nn.embedding_lookup(ctxt_word_embeddings, seq)
                one_hot = [0.0] * num_sequences
                one_hot[i] = 1.0
                mode_feature = tf.constant([[one_hot]], tf.float32)
                mode_feature = tf.tile(mode_feature, tf.stack([num_seq, max_length, 1]))
                encoded = tf.concat([encoded, mode_feature], 2)
                encoded = modular_encoder.modular_encoder(
                    sequence_module, {'text': encoded}, {'text': length}, {'text': None}, size,
                    1.0 - keep_prob, is_eval)[0]['text']

                mask = misc.mask_for_lengths(length, max_length, mask_right=False, value=1.0)
                encoded = encoded * tf.expand_dims(mask, 2)

                seq_lemmas = tf.gather(word2lemma_off, tf.reshape(seq, [-1]))
                new_lemma_embeddings = tf.unsorted_segment_max(
                    tf.reshape(encoded, [-1, size]), seq_lemmas, tf.reduce_max(word2lemma_off) + 1)
                new_lemma_embeddings = tf.nn.relu(new_lemma_embeddings)

                return tf.gather(new_lemma_embeddings, word2lemma_off)

            new_word_embeddings = tf.cond(num_seq > 0, non_zero_batchsize_op,
                                          lambda: tf.zeros_like(ctxt_word_embeddings))
            # update old word embeddings with new ones via gated addition
            gate = tf.layers.dense(tf.concat([ctxt_word_embeddings, new_word_embeddings], 1), size, tf.nn.sigmoid,
                                   bias_initializer=tf.constant_initializer(1.0), name="embeddings_gating")
            ctxt_word_embeddings = ctxt_word_embeddings * gate + (1.0 - gate) * new_word_embeddings

    return ctxt_word_embeddings, reading_sequence_offset, offsets


def word_with_char_embed(size, word_embeddings, unique_word_chars, unique_word_char_length, num_chars):
    # compute combined embeddings
    char_word_embeddings = conv_char_embedding(
        num_chars, size, unique_word_chars, unique_word_char_length)
    char_word_embeddings = tf.nn.relu(char_word_embeddings)
    gate = tf.layers.dense(tf.concat([word_embeddings, char_word_embeddings], 1), size, tf.nn.sigmoid,
                           bias_initializer=tf.constant_initializer(1.0), name="embeddings_gating")
    word_embeddings = word_embeddings * gate + (1.0 - gate) * char_word_embeddings

    return word_embeddings
