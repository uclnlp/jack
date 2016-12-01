from math import factorial
from quebap.sisyphos.batch import get_feed_dicts
from quebap.sisyphos.vocab import Vocab
from quebap.sisyphos.map import tokenize, lower, deep_map, deep_seq_map
from quebap.sisyphos.train import train
import tensorflow as tf
from quebap.sisyphos.hooks import SpeedHook, AccuracyHook, LossHook, ETAHook
from quebap.projects.storycloze.assignment3_models import get_permute_model, \
    get_basic_model
import os

def load_corpus(name, use_permutation_index=True):
    story = []
    order = []

    with open("./quebap/data/StoryCloze/%s_shuffled.tsv" % name, "r") as f:
        for line in f.readlines():
            splits = [x.strip() for x in line.split("\t")]
            current_story = splits[0:5]
            current_order = splits[5:]

            story.append(current_story)

            if use_permutation_index:
                order.append(permutation_index(current_order))
            else:
                order.append(current_order)

    return {"story": story, "order": order}


def permutation_index(p):
    result = 0
    for j in range(len(p)):
        k = sum(1 for i in p[j + 1:] if i < p[j])
        result += k * factorial(len(p) - j - 1)
    return result


def pipeline(corpus, vocab=None, target_vocab=None, emb=None, freeze=False,
             concat_seq=True, use_permutation_index=True):
    vocab = vocab or Vocab(emb=emb)
    target_vocab = target_vocab or Vocab(unk=None)
    if freeze:
        vocab.freeze()
        target_vocab.freeze()

    corpus_tokenized = deep_map(corpus, tokenize, ["story"])
    corpus_lower = deep_seq_map(corpus_tokenized, lower, ["story"])
    corpus_os = deep_seq_map(corpus_lower,
                             lambda xs: ["<SOS>"] + xs + ["<EOS>"], ["story"])
    corpus_ids = deep_map(corpus_os, vocab, ["story"])
    corpus_ids = deep_map(corpus_ids, target_vocab, ["order"])
    if concat_seq:
        for i in range(len(corpus_ids["story"])):
            corpus_ids["story"][i] = [x for xs in corpus_ids["story"][i] for x in xs]

    seq_keys = ["story"]
    if not use_permutation_index:
        seq_keys += ["order"]

    corpus_ids = deep_seq_map(corpus_ids,
                              lambda xs: len(xs), keys=seq_keys,
                              fun_name='length', expand=True)
    return corpus_ids, vocab, target_vocab


if __name__ == '__main__':
    # Config
    DEBUG = False
    USE_PERMUTATION_INDEX = False

    INPUT_SIZE = 300
    OUTPUT_SIZE = 300
    LAYERS = 1

    DROPOUT = 0.1
    L2 = 0.001
    CLIP_NORM = 5.0

    LEARNING_RATE = 0.001
    MAX_EPOCHS = 100
    BATCH_SIZE = 8 if DEBUG else 256

    # get_model = get_permute_model
    get_model = get_basic_model

    if DEBUG:
        train_corpus = load_corpus("debug", USE_PERMUTATION_INDEX)
        dev_corpus = load_corpus("debug", USE_PERMUTATION_INDEX)
        test_corpus = load_corpus("debug", USE_PERMUTATION_INDEX)
    else:
        train_corpus = load_corpus("train", USE_PERMUTATION_INDEX)
        dev_corpus = load_corpus("dev", USE_PERMUTATION_INDEX)
        test_corpus = load_corpus("test", USE_PERMUTATION_INDEX)

    train_mapped, train_vocab, train_target_vocab = \
        pipeline(train_corpus, use_permutation_index=USE_PERMUTATION_INDEX)

    train_vocab = train_vocab.prune(5)

    dev_mapped, _, _ = \
        pipeline(dev_corpus, train_vocab, train_target_vocab,
                 use_permutation_index=USE_PERMUTATION_INDEX, freeze=True)
    test_mapped, _, _ = \
        pipeline(test_corpus, train_vocab, train_target_vocab,
                 use_permutation_index=USE_PERMUTATION_INDEX, freeze=True)

    loss, placeholders, predict = \
        get_model(len(train_vocab), INPUT_SIZE, OUTPUT_SIZE,
                  len(train_target_vocab), DROPOUT)

    print("Dev Example:")
    for key in dev_mapped:
        print(key, dev_mapped[key][1])

    print("""
input vocab:  %d
target vocab: %d

train:        %d
dev:          %d
test:         %d
    """ % (len(train_vocab), len(train_target_vocab),
           len(train_mapped["story"]), len(dev_mapped["story"]),
           len(test_mapped["story"])))

    # from operator import itemgetter
    # sym2freqs = [(v, k) for k, v in train_vocab.sym2freqs.items()]
    # sym2freqs.sort(key=itemgetter(0))
    # sym2freqs = sym2freqs[::-1]
    # for sym, freq in sym2freqs[10000:10100]:
    #     print(sym, freq)

    # Training
    train_feed_dicts = get_feed_dicts(train_mapped, placeholders, BATCH_SIZE)
    dev_feed_dicts = get_feed_dicts(dev_mapped, placeholders, BATCH_SIZE)
    test_feed_dicts = get_feed_dicts(test_mapped, placeholders, BATCH_SIZE)

    optim = tf.train.AdamOptimizer(LEARNING_RATE)

    hooks = [
        LossHook(100, BATCH_SIZE),
        SpeedHook(100, BATCH_SIZE),
        ETAHook(100, MAX_EPOCHS, 500),
        AccuracyHook(train_feed_dicts, predict, placeholders['order'], 2),
        AccuracyHook(dev_feed_dicts, predict, placeholders['order'], 2),
        AccuracyHook(test_feed_dicts, predict, placeholders['order'], 2)
    ]

    train(loss, optim, train_feed_dicts, max_epochs=MAX_EPOCHS, hooks=hooks,
          l2=L2, clip=CLIP_NORM, clip_op=tf.clip_by_norm)
