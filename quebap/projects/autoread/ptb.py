import tensorflow as tf
import numpy as np
from quebap.projects.autoread.autoreader import AutoReader
from tensorflow.models.rnn.ptb import reader

if __name__ == '__main__':
    data_path = "./quebap/data/PTB/"

    batch_size = 10
    max_seq_length = 20

    train_data, valid_data, test_data, vocab_size = \
        reader.ptb_raw_data(data_path)

    train_iterator = reader.ptb_iterator(train_data, batch_size, max_seq_length)

    input_size = 10
    hidden_size = 10
    seq_lengths = [max_seq_length] * batch_size

    autoreader = AutoReader(input_size, vocab_size, max_seq_length,
                            noise=0.0, cloze_noise=1.0,
                            learning_rate=0.01,
                            forward_only=True)

    outputs = autoreader.outputs
    logits = autoreader.logits
    unk_mask = autoreader.unk_mask
    symbols = autoreader.symbols * tf.reshape(
        tf.cast(autoreader.unk_mask, tf.int64), tf.shape(autoreader.symbols)
    )
    loss = autoreader.loss

    optim_op = autoreader.update

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for i in range(1000):
            for j, inputs in train_iterator:
                batch = (inputs, seq_lengths)
                _, loss_current, symbols_current = \
                    autoreader.run(sess, [optim_op, loss, symbols], batch)
                #print("inputs:\n%s\n\nlengths:\n%s\n\n\nsymbols:\n%s\n\n%5d loss: %.3f\n\n" %
                #      (str(inputs), str(seq_lengths), str(symbols_current), i, loss_current))
                print(loss_current)
