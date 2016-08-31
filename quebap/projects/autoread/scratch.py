import tensorflow as tf
import numpy as np
from quebap.projects.autoread.autoreader import AutoReader

if __name__ == '__main__':
    input_size = 10
    hidden_size = 10
    vocab_size = 50
    unk_id = vocab_size-1
    vocab_size_full = vocab_size * 1.1
    batch_size = 10
    max_seq_length = 10

    # [mb x seq_length]
    inputs = np.random.randint(0, vocab_size_full,
                               size=(batch_size, max_seq_length))
    inputs = np.minimum(inputs, unk_id)
    seq_lengths = np.random.randint(2, max_seq_length + 1, batch_size)

    inputs_sliced = tf.slice(inputs, (0, 0), tf.pack(
        [-1, tf.cast(tf.reduce_max(seq_lengths), tf.int32)]
    ))

    autoreader = AutoReader(input_size, vocab_size, max_seq_length,
                            noise=0.0, cloze_noise=0.0,
                            learning_rate=0.01, unk_id=unk_id,
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
            batch = (inputs, seq_lengths)
            _, loss_current, symbols_current = \
                autoreader.run(sess, [optim_op, loss, symbols], batch)
            print("inputs:\n%s\n\nlengths:\n%s\n\n\nsymbols:\n%s\n\n%5d loss: %.3f\n\n" %
                  (str(inputs), str(seq_lengths), str(symbols_current), i, loss_current))

