import json
from pprint import pprint

from sisyphos.batch import get_feed_dicts, augment_with_length
from sisyphos.io import map_corpus, tokenize, lower, corpus_to_ids, \
    seqs_to_words, seqs_to_ids, Vocab, seq_to_ids
from sisyphos.models import conditional_reader_model
from sisyphos.train import train
import tensorflow as tf
import numpy as np
from sisyphos.hooks import SpeedHook, AccuracyHook, LossHook

target_vocab = Vocab({"contradiction": 0, "neutral": 1, "entailment": 2},
                     ["contradiction", "neutral", "entailment"])


def load(path, max_count=None):
    seq1s = []
    seq2s = []
    targets = []
    count = 0
    with open(path, "r") as f:
        for line in f.readlines():
            if max_count is None or count < max_count:
                instance = json.loads(line.strip())
                sentence1 = instance['sentence1']
                sentence2 = instance['sentence2']
                target = instance['gold_label']
                if target != "-":
                    seq1s.append(sentence1)
                    seq2s.append(sentence2)
                    targets.append(target)
                    count += 1
    print("Loaded %d examples from %s" % (len(targets), path))
    return [seq1s, seq2s, targets]


def pipeline(corpus, vocab=None, freeze=False):
    # not tokenizing labels
    corpus_tokenized = map_corpus(corpus, tokenize, [0, 1])
    corpus_lower = map_corpus(corpus_tokenized, lower, [0, 1])
    corpus_sos = map_corpus(corpus_lower, lambda xs: ["<SOS>"] + xs, [0, 1])
    corpus_eos = map_corpus(corpus_sos, lambda xs: xs + ["<EOS>"], [0, 1])
    corpus_ids, vocab = corpus_to_ids(corpus_eos, [0, 1], vocab, freeze)

    target_ids, _ = seq_to_ids(corpus_ids[2], target_vocab, freeze)
    corpus_ids[2] = np.asarray(target_ids)
    corpus_ids = augment_with_length(corpus_ids, [0, 1])
    return corpus_ids, vocab


if __name__ == '__main__':
    DEBUG = True

    if DEBUG:
        train_data = load("./data/snli/snli_1.0/snli_1.0_train.jsonl", 8)
        dev_data = train_data
        test_data = train_data
    else:
        train_data, dev_data, test_data = [
            load("./data/snli/snli_1.0/snli_1.0_%s.jsonl" % name)
            for name in ["train", "dev", "test"]]

    train_data, train_vocab = pipeline(train_data)
    dev_data, _ = pipeline(dev_data, train_vocab, freeze=True)
    test_data, _ = pipeline(test_data, train_vocab, freeze=True)

    input_size = 100
    output_size = 100
    vocab_size = len(train_vocab[0])
    target_size = len(target_vocab[0])
    batch_size = 2

    print("vocab size: %d" % vocab_size)

    (logits, loss, predict), placeholders = \
        conditional_reader_model(input_size, output_size, vocab_size,
                                 target_size)

    train_feed_dicts = \
        get_feed_dicts(train_data, placeholders, batch_size)
    dev_feed_dicts = \
        get_feed_dicts(dev_data, placeholders, 1)

    optim = tf.train.AdamOptimizer()


    def report_loss(sess, epoch, iter, predict, loss):
        if iter > 0 and iter % 2 == 0:
            print("epoch %4d\titer %4d\tloss %4.2f" % (epoch, iter, loss))


    hooks = [
        #report_loss,
        LossHook(10, batch_size),
        SpeedHook(100, batch_size),
        AccuracyHook(dev_feed_dicts, predict, placeholders[-1], 10)
    ]

    train(loss, optim, train_feed_dicts, max_epochs=1000, hooks=hooks)
