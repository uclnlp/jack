# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os

import numpy as np

dir = "./jack/data/emoji2vec/"
emojis = []
vecs = []
with open(dir + "metadata.tsv", "w") as f_out:
    # f_out.write("emoji\n")
    with open(dir + "emoji2vec.txt", "r") as f_in:
        for ix, line in enumerate(f_in.readlines()[1:]):
            splits = line.strip().split(" ")
            emoji = splits[0]
            vec = [float(x) for x in splits[1:]]
            assert len(vec) == 300
            # print(emoji, vec)
            emojis.append(emoji)
            vecs.append(vec)
            f_out.write(emoji+"\n")
        f_in.close()
    f_out.close()

emoji2vec = tf.constant(np.array(vecs))
tf_emoji2vec = tf.get_variable("emoji2vec", [len(vecs), 300], tf.float64)

# save embeddings to file
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf_emoji2vec.assign(emoji2vec))

    saver = tf.train.Saver()
    saver.save(sess, os.path.join(dir, "model.ckpt"), 0)

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(dir)

    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = tf_emoji2vec.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(dir, 'metadata.tsv')

    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)
