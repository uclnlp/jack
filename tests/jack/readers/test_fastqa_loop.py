# -*- coding: utf-8 -*-

import tensorflow as tf

from jack.core import SharedResources
from jack.core.tensorflow import TFReader
from jack.core.tensorport import Ports
from jack.io.embeddings.embeddings import load_embeddings
from jack.io.load import load_jack
from jack.readers.extractive_qa.shared import XQAInputModule, XQAOutputModule
from jack.readers.extractive_qa.tensorflow.fastqa import FastQAModule
from jack.util.vocab import Vocab


def test_fastqa():
    tf.reset_default_graph()

    data = load_jack('tests/test_data/squad/snippet_jtr.json')

    # fast qa must be initialized with existing embeddings, so we create some
    embeddings = load_embeddings('./tests/test_data/glove.840B.300d_top256.txt', 'glove')

    # we need a vocabulary (with embeddings for our fastqa_reader, but this is not always necessary)
    vocab = Vocab(vocab=embeddings.vocabulary)

    # ... and a config
    config = {
        "batch_size": 1,
        "repr_dim": 10,
        "with_char_embeddings": True
    }

    # create/setup reader
    shared_resources = SharedResources(vocab, config, embeddings)

    input_module = XQAInputModule(shared_resources)
    model_module = FastQAModule(shared_resources)
    output_module = XQAOutputModule()

    reader = TFReader(shared_resources, input_module, model_module, output_module)
    reader.setup_from_data(data, is_training=True)

    loss = reader.model_module.tensors[Ports.loss]
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    min_op = optimizer.minimize(loss)

    session = model_module.tf_session
    session.run(tf.global_variables_initializer())

    for epoch in range(0, 10):
        for batch in reader.input_module.batch_generator(data, 1, False):
            feed_dict = reader.model_module.convert_to_feed_dict(batch)
            loss_value, _ = session.run((loss, min_op), feed_dict=feed_dict)
            print(loss_value)
