import json
from pprint import pprint
from time import time, sleep
from os import path

from sisyphos.batch import get_feed_dicts
from sisyphos.map import tokenize, lower, Vocab, deep_map, deep_seq_map, VocabEmb
from sisyphos.models import conditional_reader_model, create_embeddings
from sisyphos.train import train
from sisyphos.examples.prepare_embeddings import load as loads_embeddings
import tensorflow as tf
import numpy as np
import random
from sisyphos.hooks import SpeedHook, AccuracyHook, LossHook


tf.set_random_seed(1337)



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


def pipeline(corpus, vocab=None, target_vocab=None, emb=None, freeze=False):
    vocab = vocab or VocabEmb(emb=emb)
    target_vocab = target_vocab or Vocab(unk=None)
    if freeze:
        vocab.freeze()
        target_vocab.freeze()

    corpus_tokenized = deep_map(corpus, tokenize, [0, 1])
    corpus_lower = deep_seq_map(corpus_tokenized, lower, [0, 1])
    corpus_sos = deep_seq_map(corpus_lower, lambda xs: ["<SOS>"] + xs, [0, 1])
    corpus_eos = deep_seq_map(corpus_sos, lambda xs: xs + ["<EOS>"], [0, 1])
    corpus_ids = deep_map(corpus_eos, vocab, [0, 1])
    corpus_ids = deep_map(corpus_ids, target_vocab, [2])
    corpus_ids = deep_seq_map(corpus_ids, lambda xs: len(xs), [0, 1], expand=True)
    corpus_ids = deep_map(corpus_ids, vocab.normalize, [0, 2]) #needs freezing next time to be comparable with other pipelines
    return corpus_ids, vocab, target_vocab


if __name__ == '__main__':
    DEBUG = True
    DEBUG_EXAMPLES = 500


    input_size = 100
    output_size = 100
    batch_size = 64
    dev_batch_size = 64 #faster than 1
    pretrain = True #use pretrained embeddings
    retrain = False #False: fix pre-trained embeddings

    learning_rate = 0.0003

    bucket_order = (0,2) #composite buckets; first over premises, then over hypotheses
    bucket_structure = (4,4) #will result in 16 composite buckets, evenly spaced over premises and hypothesis


    if DEBUG:
        train_data = load("./data/snli/snli_1.0/snli_1.0_train.jsonl", DEBUG_EXAMPLES)
        dev_data = train_data
        test_data = train_data
        if pretrain:
            emb_file = 'GoogleNews-vectors-negative300.bin'
            embeddings = loads_embeddings(path.join('quebap', 'data', 'SG_GoogleNews', emb_file), format='word2vec_bin', save=False)
#            emb_file = 'glove.6B.50d.pkl'
#            embeddings = loads_embeddings(path.join('quebap', 'data', 'GloVe', emb_file))
            print('loaded pre-trained embeddings')
        # embeddings = loads_embeddings(path.join('quebap','data','GloVe',emb_file), 'glove', {'vocab_size':400000,'dim':50})
    else:
        train_data, dev_data, test_data = [
            load("./data/snli/snli_1.0/snli_1.0_%s.jsonl" % name)
            for name in ["train", "dev", "test"]]
        if pretrain:
            emb_file = 'GoogleNews-vectors-negative300.bin'
            embeddings = loads_embeddings(path.join('quebap', 'data', 'SG_GoogleNews', emb_file), format='word2vec_bin', save=False)

    #load pre-trained embeddings

    emb = embeddings.get if pretrain else None

    train_data, train_vocab, train_target_vocab = pipeline(train_data, emb=emb)
    N_oov = train_vocab.count_oov()
    N_pre = train_vocab.count_pretrained()
    print('In Training data vocabulary: %d pre-trained, %d out-of-vocab.'%(N_pre,N_oov))

    dev_data, _, _ = pipeline(dev_data, train_vocab, train_target_vocab,
                              freeze=True)
    test_data, _, _ = pipeline(test_data, train_vocab, train_target_vocab,
                               freeze=True)


    embeddings_matrix = create_embeddings(train_vocab, retrain=retrain)
    # todo: transform longer embeddings to input_size if they are fixed.
    # todo: Would be faster than automatically doing it in embedder (needed in case trainable)


    vocab_size = len(train_vocab)
    target_size = len(train_target_vocab)

    print("vocab size:  %d" % vocab_size)
    print("target size: %d" % target_size)

    (logits, loss, predict), placeholders = \
        conditional_reader_model(input_size, output_size, vocab_size,
                                 target_size, embeddings=embeddings_matrix)

    train_feed_dicts = \
        get_feed_dicts(train_data, placeholders, batch_size,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)
    dev_feed_dicts = \
        get_feed_dicts(dev_data, placeholders, dev_batch_size,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)

    optim = tf.train.AdamOptimizer(learning_rate)


    def report_loss(sess, epoch, iter, predict, loss):
        if iter > 0 and iter % 2 == 0:
            print("epoch %4d\titer %4d\tloss %4.2f" % (epoch, iter, loss))


    hooks = [
        #report_loss,
        LossHook(100, batch_size),
        SpeedHook(100, batch_size),
        AccuracyHook(dev_feed_dicts, predict, placeholders[-1], 2)
    ]

    train(loss, optim, train_feed_dicts, max_epochs=1000, hooks=hooks)

    #TODO: evaluate on test data


    print('finished in %.3fh'%((time()-t0)/3600.))

