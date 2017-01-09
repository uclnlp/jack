import json
from pprint import pprint
from time import time, sleep
from os import path

from quebap.sisyphos.batch import get_feed_dicts
from quebap.sisyphos.vocab import Vocab, NeuralVocab
from quebap.sisyphos.map import tokenize, lower, deep_map, deep_seq_map
from quebap.sisyphos.models import conditional_reader_model
from quebap.sisyphos.train import train
from quebap.io.embeddings.embeddings import load_embeddings
import tensorflow as tf
import numpy as np
import random
from quebap.sisyphos.hooks import ExamplesPerSecHook, AccuracyHook, LossHook


tf.set_random_seed(1337)

from time import time


class Duration(object):
    def __init__(self):
        self.t0 = time()
        self.t = time()
    def __call__(self):
        print('Time since last checkpoint : %.2fmin'%((time()-self.t)/60.))
        self.t = time()

checkpoint = Duration()



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
    return {'sentence1': seq1s, 'sentence2': seq2s, 'targets': targets}


def pipeline(corpus, vocab=None, target_vocab=None, emb=None, freeze=False, normalize=False):
    vocab = vocab or Vocab(emb=emb)
    target_vocab = target_vocab or Vocab(unk=None)
    if freeze:
        vocab.freeze()
        target_vocab.freeze()

    corpus_tokenized = deep_map(corpus, tokenize, ['sentence1', 'sentence2'])
    corpus_lower = deep_seq_map(corpus_tokenized, lower, ['sentence1', 'sentence2'])
    corpus_os = deep_seq_map(corpus_lower, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ['sentence1', 'sentence2'])
    corpus_ids = deep_map(corpus_os, vocab, ['sentence1', 'sentence2'])
    corpus_ids = deep_map(corpus_ids, target_vocab, ['targets'])
    corpus_ids = deep_seq_map(corpus_ids, lambda xs: len(xs), keys=['sentence1', 'sentence2'], fun_name='lengths', expand=True)
    if normalize:
        corpus_ids = deep_map(corpus_ids, vocab._normalize, keys=['sentence1', 'sentence2'])
    return corpus_ids, vocab, target_vocab


if __name__ == '__main__':

    t0 = time()
    DEBUG = True
    DEBUG_EXAMPLES = 2000

    ATTENTIVE = False

    input_size = 100
    output_size = 100
    batch_size = 256
    dev_batch_size = 256
    pretrain = True #use pretrained embeddings
    retrain = True #False: fix pre-trained embeddings

    learning_rate = 0.002

    bucket_order = ('sentence1', 'sentence2') #composite buckets; first over premises, then over hypotheses
    bucket_structure = (4, 4) #will result in 16 composite buckets, evenly spaced over premises and hypothesis

    if DEBUG:
        train_data = load("./quebap/data/SNLI/snli_1.0/snli_1.0_train.jsonl", DEBUG_EXAMPLES)
        dev_data = train_data
        test_data = train_data
    else:
        train_data, dev_data, test_data = [load("./quebap/data/SNLI/snli_1.0/snli_1.0_%s.jsonl" % name)\
                                           for name in ["train", "dev", "test"]]

    print(train_data)

    print('loaded train/dev/test data')
    if pretrain:
        if DEBUG:
            emb_file = 'glove.6B.50d.txt'
            embeddings = load_embeddings(path.join('quebap', 'data', 'GloVe', emb_file),'glove')
        else:
            #emb_file = 'GoogleNews-vectors-negative300.bin.gz'
            #embeddings = load_embeddings(path.join('quebap', 'data', 'word2vec', emb_file),'word2vec')
            emb_file = 'glove.840B.300d.zip'
            embeddings = load_embeddings(path.join('quebap', 'data', 'GloVe', emb_file),'glove')
        print('loaded pre-trained embeddings')

    emb = embeddings.get if pretrain else None

    checkpoint()
    print('encode train data')
    train_data, train_vocab, train_target_vocab = pipeline(train_data, emb=emb, normalize=True)


    N_oov = train_vocab.count_oov()
    N_pre = train_vocab.count_pretrained()
    print('In Training data vocabulary: %d pre-trained, %d out-of-vocab.'%(N_pre,N_oov))

    vocab_size = len(train_vocab)
    target_size = len(train_target_vocab)

    print("\tvocab size:  %d" % vocab_size)
    print("\ttarget size: %d" % target_size)

    checkpoint()
    print('encode dev data')
    dev_data, _, _ = pipeline(dev_data, train_vocab, train_target_vocab,
                              freeze=True)
    checkpoint()
    print('encode test data')
    test_data, _, _ = pipeline(test_data, train_vocab, train_target_vocab,
                               freeze=True)
    checkpoint()


    print('build NeuralVocab')
    nvocab = NeuralVocab(train_vocab, input_size=input_size, use_pretrained=True, train_pretrained=False, unit_normalize=True)


    checkpoint()
    print('build model')
    (logits, loss, predict), placeholders = \
        conditional_reader_model(output_size, target_size, nvocab, attentive=ATTENTIVE)

    train_feed_dicts = \
        get_feed_dicts(train_data, placeholders, batch_size,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)
    dev_feed_dicts = \
        get_feed_dicts(dev_data, placeholders, dev_batch_size,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)

    optim = tf.train.AdamOptimizer(learning_rate)


    #for i, dict in enumerate(train_feed_dicts):
    #    if i == 0:
    #        print(dict)
    #    else:
    #        pass


    def report_loss(sess, epoch, iter, predict, loss):
        if iter > 0 and iter % 2 == 0:
            print("epoch %4d\titer %4d\tloss %4.2f" % (epoch, iter, loss))


    hooks = [
        #report_loss,
        LossHook(100, batch_size),
        ExamplesPerSecHook(100, batch_size),
        AccuracyHook(dev_feed_dicts, predict, placeholders['targets'], 2)
    ]

    train(loss, optim, train_feed_dicts, max_epochs=1000, hooks=hooks)

    #TODO: evaluate on test data
    print('finished in %.3fh'%((time()-t0)/3600.))

