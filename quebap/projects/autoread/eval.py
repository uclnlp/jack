import json
from quebap.projects.autoread.autoreader import AutoReader
from quebap.tensorizer import GenericTensorizer
import tensorflow as tf
import numpy as np


def load_vocab(path, max_vocab_size=50000):
    vocab = {}
    with open(path, "r") as f:
        for line in f.readlines()[2:max_vocab_size]:
            splits = line.split("\t")
            vocab[splits[1]] = int(splits[0])
    return vocab


def vocab_to_ixmap(vocab):
    ixmap = {}
    for word in vocab:
        ixmap[vocab[word]] = word
    return ixmap


def reindex_seq(seq, source_vocab_ixmap, target_vocab):
    """
    :param seq: [batch_size x max_seq_length] int32 word ids
    :param source_vocab_ixmap: dictionary mapping source word ids to words
    :param target_vocab: dictionary mapping targit words to ids
    :return: [batch_size x max_seq_length] int32 with target word ids
    """
    for row in range(len(seq)):
        for col in range(len(seq[0])):
            word = source_vocab_ixmap[seq[row][col]]
            if word not in target_vocab:
                word = "<UNK>"

            seq[row][col] = target_vocab[word]


if __name__ == '__main__':
    with open("./quebap/data/LS/debug/lexsub_debug_cleaned.jsonl", "r") as f:
        data = json.load(f)
    tensorizer = GenericTensorizer(data)

    quebap_vocab = tensorizer.question_lexicon
    quebap_vocab_ixmap = vocab_to_ixmap(quebap_vocab)
    vocab = load_vocab("./quebap/projects/autoread/document.vocab")

    config = {
        "size": 3,  # todo
        "vocab_size": len(vocab),
        "is_train": False,
    }

    batch_size = 1

    reader = AutoReader.create_from_config(config)

    outputs = reader.outputs

    # todo: should be entire vocab
    candidate_representations = None

    # should be language model
    logits = None

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for quebap_batch in tensorizer.create_batches(data, batch_size=batch_size):
            # todo: reindex words
            seq = quebap_batch[tensorizer.questions]
            reindex_seq(seq, quebap_vocab_ixmap, vocab)

            batch = [seq, quebap_batch[tensorizer.question_lengths],
                     np.ones((batch_size,
                              len(quebap_batch[tensorizer.questions][0])))]

            results = reader.run(sess, [outputs], batch)

            print(results)
