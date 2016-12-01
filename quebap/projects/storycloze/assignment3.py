from math import factorial
from quebap.sisyphos.batch import get_feed_dicts
from quebap.sisyphos.vocab import Vocab
from quebap.sisyphos.map import tokenize, lower, deep_map, deep_seq_map
from quebap.sisyphos.train import train
import tensorflow as tf
from quebap.sisyphos.hooks import SpeedHook, AccuracyHook, LossHook, ETAHook


def load_corpus(name):
    story = []
    order = []

    with open("./quebap/data/StoryCloze/%s_shuffled.tsv" % name, "r") as f:
        for line in f.readlines():
            splits = [x.strip() for x in line.split("\t")]
            current_story = splits[0:5]
            current_order = splits[5:]

            story.append(current_story)
            order.append(permutation_index(current_order))

    return {"story": story, "order": order}


def permutation_index(p):
    result = 0
    for j in range(len(p)):
        k = sum(1 for i in p[j + 1:] if i < p[j])
        result += k * factorial(len(p) - j - 1)
    return result


def pipeline(corpus, vocab=None, target_vocab=None, emb=None, freeze=False,
             concat_seq=True):
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
    corpus_ids = deep_seq_map(corpus_ids,
                              lambda xs: len(xs), keys=["story"],
                              fun_name='length', expand=True)
    return corpus_ids, vocab, target_vocab


def get_model(vocab_size, input_size, output_size, target_size, dropout=0.0):
    # Placeholders
    # [batch_size x max_length]
    story = tf.placeholder(tf.int64, [None, None], "story")
    # [batch_size]
    story_length = tf.placeholder(tf.int64, [None], "story_length")
    # [batch_size]
    order = tf.placeholder(tf.int64, [None], "order")
    placeholders = {"story": story, "story_length": story_length, "order": order}

    # Word embeddings
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    embeddings = tf.get_variable("W", [vocab_size, input_size],
                                 initializer=initializer)
    # [batch_size x max_seq_length x input_size]
    story_embedded = tf.nn.embedding_lookup(embeddings, story)

    with tf.variable_scope("reader") as varscope:
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        cell_dropout = \
            tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0-dropout)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_dropout,
            cell_dropout,
            story_embedded,
            sequence_length=story_length,
            dtype=tf.float32
        )

        c, h = states[-1]  # LSTM state is a tuple

        logits = tf.contrib.layers.linear(h, target_size)

        predict = tf.arg_max(tf.nn.softmax(logits), 1)

        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order))

        return loss, placeholders, predict


if __name__ == '__main__':
    # Config
    DEBUG = False
    INPUT_SIZE = 300
    OUTPUT_SIZE = 300
    BATCH_SIZE = 512  # 8

    DROPOUT = 0.1
    L2 = 0.001
    CLIP_NORM = 5.0
    LEARNING_RATE = 0.01
    MAX_EPOCHS = 100

    if DEBUG:
        train_corpus = load_corpus("debug")
        dev_corpus = load_corpus("debug")
        test_corpus = load_corpus("debug")
    else:
        train_corpus = load_corpus("train")
        dev_corpus = load_corpus("dev")
        test_corpus = load_corpus("test")

    train_mapped, train_vocab, train_target_vocab = pipeline(train_corpus)
    dev_mapped, _, _ = pipeline(dev_corpus, train_vocab, train_target_vocab)
    test_mapped, _, _ = pipeline(test_corpus, train_vocab, train_target_vocab)

    loss, placeholders, predict = \
        get_model(len(train_vocab), INPUT_SIZE, OUTPUT_SIZE,
                  len(train_target_vocab), DROPOUT)

    # Training
    train_feed_dicts = get_feed_dicts(train_mapped, placeholders, BATCH_SIZE)
    dev_feed_dicts = get_feed_dicts(dev_mapped, placeholders, BATCH_SIZE)
    test_feed_dicts = get_feed_dicts(test_mapped, placeholders, BATCH_SIZE)

    optim = tf.train.AdamOptimizer(LEARNING_RATE)

    hooks = [
        LossHook(100, BATCH_SIZE),
        SpeedHook(100, BATCH_SIZE),
        ETAHook(100, MAX_EPOCHS, 500),
        AccuracyHook(train_feed_dicts, predict, placeholders['order'], 10),
        AccuracyHook(dev_feed_dicts, predict, placeholders['order'], 2)
    ]

    train(loss, optim, train_feed_dicts, max_epochs=MAX_EPOCHS, hooks=hooks,
          l2=L2, clip=CLIP_NORM, clip_op=tf.clip_by_norm)
