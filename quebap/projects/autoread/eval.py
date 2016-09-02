import json
from quebap.projects.autoread.autoreader import AutoReader
from quebap.projects.autoread.util import init_with_word_embeddings
from quebap.tensorizer import GenericTensorizer
import tensorflow as tf
import numpy as np

PLACEHOLDER = "XXXXX"

"""HACK

"""


def load_vocab(path, max_vocab_size=50000):
    vocab = {}
    with open(path, "r") as f:
        for line in f.readlines()[2:max_vocab_size]:
            splits = line.split("\t")
            vocab[splits[1]] = int(splits[0])
    vocab[PLACEHOLDER] = len(vocab)
    return vocab


def reindex_seq(seq, source_vocab_ixmap, target_vocab):
    """
    :param seq: [batch_size x max_seq_length] int32 word ids
    :param source_vocab_ixmap: dictionary mapping source word ids to words
    :param target_vocab: dictionary mapping targit words to ids
    """
    for row in range(len(seq)):
        for col in range(len(seq[0])):
            word = source_vocab_ixmap[seq[row][col]]
            if word not in target_vocab:
                word = "<UNK>"

            seq[row][col] = target_vocab[word]


def seq_to_symbols(seq, vocab_ixmap):
    """
    :param seq: [batch_size x max_seq_length] int32 word ids
    :param vocab_ixmap: dictionary mapping source word ids to words
    :return: [batch_size x max_seq_length] words as strings
    """
    return [[vocab_ixmap[seq[row][col]] for row in range(len(seq))]
            for col in range(len(seq[0]))]

if __name__ == '__main__':
    with open("./quebap/data/LS/debug/lexsub_debug_cleaned.jsonl", "r") as f:
        data = json.load(f)
    tensorizer = GenericTensorizer(data)

    # quebap_vocab = tensorizer.question_lexicon
    quebap_vocab = tensorizer.support_lexicon
    quebap_vocab_ixmap = AutoReader.vocab_to_ixmap(quebap_vocab)
    vocab = AutoReader.load_vocab()
    vocab_ixmap = AutoReader.vocab_to_ixmap(vocab)

    config = {
        "size": 300,
        "vocab_size": len(vocab),
        "is_train": False,
        "word_embeddings": ""  # "word2vec"
    }

    batch_size = 1
    k = 5

    reader = AutoReader.create_from_config(config, word_embeddings=config["word_embeddings"])
    outputs = reader.outputs
    logits = reader.logits

    # todo: add op for getting k-best decoded symbols

    top_k = tf.nn.top_k(logits, k)

    HACK = True

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        init_with_word_embeddings(sess, reader)
        if HACK:
            # sum of word vectors baseline
            reader.outputs = \
                tf.tile(tf.reduce_mean(
                    reader.outputs, 1, keep_dims=True
                ), tf.pack([1, tf.cast(tf.reduce_max(
                    reader._seq_lengths), tf.int32), 1]))

        for quebap_batch in tensorizer.create_batches(data, batch_size=batch_size):
            question_word_position = quebap_batch[tensorizer.questions][0][0]
            seq = quebap_batch[tensorizer.support][0]
            reindex_seq(seq, quebap_vocab_ixmap, vocab)
            batch = [seq, quebap_batch[tensorizer.support_indices][0],
                     np.ones((batch_size,
                              len(quebap_batch[tensorizer.support][0][0])))]

            results = reader.run(sess, [top_k], batch)
            seq_words = seq_to_symbols(seq, vocab_ixmap)
            top_k_words = seq_to_symbols(results[0][1][0], vocab_ixmap)

            print(quebap_batch[tensorizer.questions])
            print(question_word_position)

            for current_batch in range(batch_size):
                for ix in range(len(seq_words)):
                    seq_word = seq_words[ix][current_batch]
                    top_k_word = [top_k_words[i][ix] for i in range(len(top_k_words))]
                    # if seq_word == PLACEHOLDER:
                    if ix == question_word_position:
                        print("%s\t%s" % (seq_word, "\t".join(top_k_word)))
                    else:
                        print(seq_word)

