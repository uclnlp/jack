'''This models is an example for training a classifier on SNLI'''
from __future__ import print_function

import os
import sys
import urllib
import zipfile
from os.path import join

import numpy as np
import tensorflow as tf
from jtr.util.hdf5_processing.hooks import AccuracyHook, LossHook, ETAHook
from jtr.util.hdf5_processing.pipeline import Pipeline
from jtr.util.hdf5_processing.processors import AddToVocab, CreateBinsByNestedLength, SaveLengthsToState, \
    ConvertTokenToIdx, StreamToHDF5, Tokenizer
from jtr.util.hdf5_processing.processors import JsonLoaderProcessors, DictKey2ListMapper, \
    RemoveLineOnJsonValueCondition, ToLower
from jtr.util.hdf5_processing.processors import TensorFlowConfig

from jtr.util.global_config import Config, Backends
from jtr.util.hdf5_processing.batching import StreamBatcher
from jtr.util.logger import Logger, LogLevel
from jtr.util.util import get_data_path

Config.parse_argv(sys.argv)

np.set_printoptions(suppress=True)

def download_snli():
    '''Creates data and snli paths and downloads SNLI in the home dir'''
    home = os.environ['HOME']
    data_dir = join(home, '.data')
    snli_dir = join(data_dir, 'snli')
    snli_url = 'http://nlp.stanford.edu/projects/snli/snli_1.0.zip'

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(snli_dir):
        os.mkdir(snli_dir)

    if not os.path.exists(join(data_dir, 'snli_1.0.zip')):
        print('Downloading SNLI...')
        snlidownload = urllib.URLopener()
        snlidownload.retrieve(snli_url, join(data_dir, "snli_1.0.zip"))

    print('Opening zip file...')
    archive = zipfile.ZipFile(join(data_dir, 'snli_1.0.zip'), 'r')

    return archive, snli_dir

class TFSNLI(object):
    def __init__(self, vocab, embedding_size, hidden_size):
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

    def forward(self):

        Q = TensorFlowConfig.inp
        S = TensorFlowConfig.support
        Q_len = TensorFlowConfig.input_length
        S_len = TensorFlowConfig.support_length
        t = TensorFlowConfig.target

        embeddings = tf.get_variable("embeddings", [self.vocab.num_token, self.embedding_size],
                                initializer=tf.random_normal_initializer(0., 1./np.sqrt(self.embedding_size)),
                                trainable=True, dtype="float32")

        with tf.variable_scope("embedders") as varscope:
            seqQ = tf.nn.embedding_lookup(embeddings, Q)
            varscope.reuse_variables()
            seqS = tf.nn.embedding_lookup(embeddings, S)

        with tf.variable_scope("conditional_reader_seq1") as varscope1:
            #seq1_states: (c_fw, h_fw), (c_bw, h_bw)
            _, seq1_states = self.reader(seqQ, Q_len, self.hidden_size, scope=varscope1)
        with tf.variable_scope("conditional_reader_seq2") as varscope2:
            varscope1.reuse_variables()
            # each [batch_size x max_seq_length x output_size]
            outputs, states = self.reader(seqS, S_len, self.hidden_size, seq1_states, scope=varscope2)

        output = tf.concat([states[0][1], states[1][1]], 1)

        logits, loss, predict = self.predictor(output, t, 3)

        return logits, loss, predict

    def reader(self, inputs, lengths, output_size, contexts=(None, None), scope=None):
        with tf.variable_scope(scope or "reader") as varscope:

            cell = tf.contrib.rnn.LSTMCell(output_size, state_is_tuple=True,initializer=tf.contrib.layers.xavier_initializer())

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell,
                cell,
                inputs,
                sequence_length=lengths,
                initial_state_fw=contexts[0],
                initial_state_bw=contexts[1],
                dtype=tf.float32)

            return outputs, states

    def predictor(self, inputs, targets, target_size):
        init = tf.contrib.layers.xavier_initializer(uniform=True) #uniform=False for truncated normal
        logits = tf.contrib.layers.fully_connected(inputs, target_size, weights_initializer=init, activation_fn=None)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                labels=targets), name='predictor_loss')
        predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')
        return logits, loss, predict

def preprocess_SNLI(clear_data=True):
    tokenizer = lambda x: x.split(' ')

    zip_path = join(get_data_path(), 'snli_1.0.zip', 'snli_1.0')
    file_paths = ['snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl']

    not_t = []
    t = ['input', 'support', 'target']
    # tokenize and input_output to hdf5
    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline('snli_example', clear_data)
    p.add_path(join(zip_path, file_paths[0]))
    p.add_line_processor(JsonLoaderProcessors())
    p.add_line_processor(RemoveLineOnJsonValueCondition('gold_label', lambda label: label == '-'))
    p.add_line_processor(DictKey2ListMapper(['sentence1', 'sentence2', 'gold_label']))
    p.add_sent_processor(ToLower())
    p.add_sent_processor(Tokenizer(tokenizer), t)
    p.add_token_processor(AddToVocab())
    p.add_post_processor(SaveLengthsToState())
    p.execute()
    p.clear_processors()
    p.save_vocabs()

    # 2. Process the data further to stream it to hdf5
    p.add_sent_processor(ToLower())
    p.add_sent_processor(Tokenizer(tokenizer), t)
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(CreateBinsByNestedLength('snli_train', min_batch_size=128))
    state = p.execute()

    # dev and test data
    p2 = Pipeline('snli_example')
    p2.copy_vocab_from_pipeline(p)
    p2.add_path(join(zip_path, file_paths[1]))
    p2.add_line_processor(JsonLoaderProcessors())
    p2.add_line_processor(RemoveLineOnJsonValueCondition('gold_label', lambda label: label == '-'))
    p2.add_line_processor(DictKey2ListMapper(['sentence1', 'sentence2', 'gold_label']))
    p2.add_sent_processor(ToLower())
    p2.add_sent_processor(Tokenizer(tokenizer), t)
    p2.add_post_processor(SaveLengthsToState())
    p2.execute()

    p2.clear_processors()
    p2.add_sent_processor(ToLower())
    p2.add_sent_processor(Tokenizer(tokenizer), t)
    p2.add_post_processor(ConvertTokenToIdx())
    p2.add_post_processor(StreamToHDF5('snli_dev'))
    p2.execute()

def main():
    Logger.GLOBAL_LOG_LEVEL = LogLevel.INFO
    Config.backend = Backends.TENSORFLOW
    Config.cuda = True
    Config.dropout = 0.1
    Config.hidden_size = 128
    Config.embedding_size = 256
    Config.L2 = 0.00003

    do_process = False
    if do_process:
        preprocess_SNLI(clear_data=True)


    p = Pipeline('snli_example')
    p.load_vocabs()
    vocab = p.state['vocab']['general']

    batch_size = 128
    TensorFlowConfig.init_batch_size(batch_size)
    train_batcher = StreamBatcher('snli_example', 'snli_train', batch_size, randomize=True, loader_threads=8)
    dev_batcher = StreamBatcher('snli_example', 'snli_dev', batch_size)

    train_batcher.subscribe_to_events(LossHook('Train', print_every_x_batches=100))
    train_batcher.subscribe_to_events(AccuracyHook('Train', print_every_x_batches=100))
    dev_batcher.subscribe_to_events(AccuracyHook('Dev', print_every_x_batches=100))
    dev_batcher.subscribe_to_events(LossHook('Dev', print_every_x_batches=100))
    eta = ETAHook(print_every_x_batches=100)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)

    sess = TensorFlowConfig.get_session()
    model = TFSNLI(vocab, embedding_size=256, hidden_size=128)
    optimizer = tf.train.AdamOptimizer(0.001)
    print('starting training...')

    logits, loss, predict = model.forward()

    min_op = optimizer.minimize(loss)

    tf.global_variables_initializer().run(session=sess)

    epochs = 10
    for epoch in range(epochs):
        for str2var, feed_dict in train_batcher:
            _, current_loss, argmax = sess.run([min_op, loss, predict], feed_dict=feed_dict)
            train_batcher.state.loss = current_loss
            train_batcher.state.targets = feed_dict[str2var['target']]
            train_batcher.state.argmax = argmax

    net.eval()
    for i, (str2var, feed_dict) in enumerate(dev_batcher):
        _, current_loss, argmax = sess.run([min_op, loss, predict], feed_dict=feed_dict)
        dev_batcher.state.loss = current_loss
        dev_batcher.state.targets = feed_dict[str2var['target']]
        dev_batcher.state.argmax = argmax


if __name__ == '__main__':
    main()
