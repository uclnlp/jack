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

if __name__ == '__main__':
    with open("./quebap/data/LS/debug/lexsub_debug_cleaned.jsonl", "r") as f:
        data = json.load(f)
    tensorizer = GenericTensorizer(data)

    quebap_vocab = tensorizer.question_lexicon
    vocab = load_vocab("./quebap/projects/autoread/document.vocab")
    # todo map from quebap to autoreader vocabs

    print(quebap_vocab)
    print(vocab)

    config = {
        "size": 3, # todo
        "vocab_size": 5, # todo
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
            batch = [
                quebap_batch[tensorizer.questions],
                quebap_batch[tensorizer.question_lengths],
                np.ones((batch_size,
                         len(quebap_batch[tensorizer.questions][0])))
            ]

            results = reader.run(sess, [outputs], batch)

            print(results)
