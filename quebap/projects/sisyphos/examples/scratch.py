import tensorflow as tf
import numpy as np
import random
from sisyphos.batch import augment_with_length, get_feed_dicts
from sisyphos.train import train
from sisyphos.models import conditional_reader_model

if __name__ == '__main__':
    N = 256
    vocab_size = 10000
    max_len = 20

    input_size = 100
    output_size = 100
    target_size = 3

    batch_size = 64
    num_targets = 3

    np.random.seed(1337)
    random.seed(1337)

    seq1_sampled = []
    for _ in range(0, N):
        len = np.random.randint(2, max_len)
        seq1_sampled.append(np.random.randint(1, vocab_size, [len]))

    seq2_sampled = []
    for _ in range(0, N):
        len = np.random.randint(2, max_len + 2)
        seq2_sampled.append(np.random.randint(1, vocab_size, [len]))

    targets_sampled = []
    for _ in range(0, N):
        targets_sampled.append(np.random.randint(0, num_targets, [1]))

    data = [seq1_sampled, seq2_sampled, targets_sampled]
    data = augment_with_length(data, [0, 1])

    (logits, loss, predict), placeholders = \
        conditional_reader_model(input_size, output_size, vocab_size, target_size)

    feed_dicts = \
        get_feed_dicts(data, placeholders, batch_size)

    optim = tf.train.AdamOptimizer()

    def report_loss(sess, epoch, iter, predict, loss):
        if iter > 0 and iter % 3 == 0:
            print("epoch %4d\titer %4d\tloss %4.2f" % (epoch, iter, loss))

    hooks = [
        report_loss
    ]

    train(loss, optim, feed_dicts, max_epochs=1000, hooks=hooks)
