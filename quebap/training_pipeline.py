import argparse
import json
import os.path as path
import tensorflow as tf
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


#from quebap.model.models import create_log_linear_reader, \
#    create_model_f_reader, create_bag_of_embeddings_reader, \
#    create_sequence_embeddings_reader, create_support_bag_of_embeddings_reader
#from quebap.tensorizer import *

from quebap.sisyphos.batch import get_feed_dicts
from quebap.sisyphos.vocab import Vocab, VocabEmb
from quebap.sisyphos.map import tokenize, lower, deep_map, deep_seq_map
from quebap.sisyphos.models import create_embeddings  # conditional_reader_model,
from quebap.sisyphos.train import train
from quebap.sisyphos.prepare_embeddings import load as loads_embeddings
from quebap.sisyphos.hooks import SpeedHook, AccuracyHook, LossHook
from quebap.model.models import conditional_reader_model

def sentihood_load(path, max_count=None):
    """
    Works for sentihood dataset. This will be transformed into a general-purpose loader later.
    """

    reading_dataset = json.load(path)

    seq1s = []
    seq2s = []
    targets = []
    count = 0
    for instance in reading_dataset['instances']:
        if max_count is None or count < max_count:
            sentence1 = instance['questions'][0]
            sentence2 = instance['support'][0]['text']
            target = instance['answers'][0]['text']
            if target != "-":
                seq1s.append(sentence1)
                seq2s.append(sentence2)
                targets.append(target)
                count += 1
    print("Loaded %d examples from %s" % (len(targets), path))
    return {'sentence1': seq1s, 'sentence2': seq2s, 'targets': targets}


def pipeline(corpus, vocab=None, target_vocab=None, emb=None, freeze=False):
    vocab = vocab or VocabEmb(emb=emb)
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
    corpus_ids = deep_map(corpus_ids, vocab.normalize, ['sentence1', 'sentence2']) #needs freezing next time to be comparable with other pipelines
    return corpus_ids, vocab, target_vocab


def main():
    # this is where the list of all models lives, add those if they work
    reader_models = {
        #'log_linear': create_log_linear_reader,
        #'model_f': create_model_f_reader,
        #'boe': create_bag_of_embeddings_reader
    }

    #@todo: test with different quebap files (ids, no ids, single/multi support/question per instance, candidates/no candidates, local/global candidates)
    #@todo: add decorators
    #@todo: dict with choice of different types of embeddings for pretraining
    #@todo: dict for different optimisers
    #@todo: parameters for different bucket structures
    #@todo: uncomment other options again once they work

    parser = argparse.ArgumentParser(description='Train and Evaluate a machine reader')
    parser.add_argument('--debug', default=True, type=bool, help="Run in debug mode, in which case the training file is also used for testing")
    parser.add_argument('--debug_examples', default=2000, type=int, help="If in debug mode, how many examples should be used")
    parser.add_argument('--train', default='data/sentihood/single_quebap.json', type=argparse.FileType('r'), help="Quebap training file")
    parser.add_argument('--test', default='data/sentihood/single_quebap.json', type=argparse.FileType('r'), help="Quebap test file")
    parser.add_argument('--batch_size', default=5, type=int, help="Batch size for training data")
    parser.add_argument('--dev_batch_size', default=5, type=int, help="Batch size for development data")
    parser.add_argument('--repr_dim_input', default=5, type=int, help="Size of the input representation (embeddings)")
    parser.add_argument('--repr_dim_output', default=5, type=int, help="Size of the output representation")
    parser.add_argument('--pretrain', default=False, type=bool, help="Use pretrained embeddings, by default the initialisation is random")
    parser.add_argument('--model', default='model_f', choices=sorted(reader_models.keys()), help="Reading model to use")
    parser.add_argument('--learning_rate', default=0.001, type=int, help="Learning rate")
    parser.add_argument('--epochs', default=3, type=int, help="Number of epochs to train for")
    #parser.add_argument('--train_begin', default=0, metavar='B', type=int, help="Use if training and test are the same file and the training set needs to be split. Index of first training instance.")
    #parser.add_argument('--train_end', default=-1, metavar='E', type=int,
    #                    help="Use if training and test are the same file and the training set needs to be split. Index of last training instance plus 1.")
    #parser.add_argument('--candidate_split', default="$", type=str, metavar="S",
    #                    help="Regular Expression for tokenizing candidates")
    #parser.add_argument('--question_split', default="-", type=str, metavar="S",
    #                    help="Regular Expression for tokenizing questions")
    #parser.add_argument('--support_split', default="-", type=str, metavar="S",
    #                    help="Regular Expression for tokenizing support")
    #parser.add_argument('--use_train_generator_for_test', default=False, type=bool, metavar="B",
    #                    help="Should the training candidate generator be used when testing")
    #parser.add_argument('--feature_type', default=None, type=str, metavar="F",
    #                    help="When using features: type of features.")

    args = parser.parse_args()

    # reading_dataset = shorten_reading_dataset(json.load(args.train), args.train_begin, args.train_end)

    # reader_model = reader_models[args.model](reading_dataset, **vars(args))

    # train_reader(reader_model, reading_dataset, reading_dataset, args.epochs, args.batch_size,
    #             use_train_generator_for_test=True)


    bucket_order = ('sentence1','sentence2') #composite buckets; first over premises, then over hypotheses
    bucket_structure = (4,4) #will result in 16 composite buckets, evenly spaced over premises and hypothesis

    if args.debug:
        train_data = sentihood_load(args.train, args.debug_examples)
        dev_data = train_data
        test_data = train_data
        if args.pretrain:
            emb_file = 'glove.6B.50d.pkl'
            embeddings = loads_embeddings(path.join('quebap', 'data', 'GloVe', emb_file))
            # embeddings = loads_embeddings(path.join('quebap','data','GloVe',emb_file), 'glove', {'vocab_size':400000,'dim':50})
    else:
        train_data, dev_data, test_data = [sentihood_load("./data/snli/snli_1.0/snli_1.0_%s.jsonl" % name) \
                                           for name in ["train", "dev", "test"]]
        print('loaded train/dev/test data')
        if args.pretrain:
            emb_file = 'GoogleNews-vectors-negative300.bin'
            embeddings = loads_embeddings(path.join('quebap', 'data', 'SG_GoogleNews', emb_file), format='word2vec_bin',
                                          save=False)
            print('loaded pre-trained embeddings')

    checkpoint()
    print('encode train data')
    train_data, train_vocab, train_target_vocab = pipeline(train_data)  # , emb=emb)
    N_oov = train_vocab.count_oov()
    N_pre = train_vocab.count_pretrained()
    print('In Training data vocabulary: %d pre-trained, %d out-of-vocab.' % (N_pre, N_oov))

    vocab_size = len(train_vocab)
    target_size = len(train_target_vocab)

    # @todo: we should allow to set vocab_size for smaller vocab
    parser.add_argument('--vocab_size', default=vocab_size, type=int)
    parser.add_argument('--target_size', default=target_size, type=int)

    args = parser.parse_args()

    #args["vocab_size"] = vocab_size
    #args["target_size"] = target_size

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

    print('create embeddings matrix')
    embeddings_matrix = create_embeddings(train_vocab, retrain=True) if args.pretrain else None

    checkpoint()
    print('build model')
    (logits, loss, predict), placeholders = conditional_reader_model(embeddings_matrix, **vars(args))

    train_feed_dicts = \
        get_feed_dicts(train_data, placeholders, args.batch_size,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)
    dev_feed_dicts = \
        get_feed_dicts(dev_data, placeholders, args.dev_batch_size,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)

    optim = tf.train.AdamOptimizer(args.learning_rate)

    hooks = [
        # report_loss,
        LossHook(100, args.batch_size),
        SpeedHook(100, args.batch_size),
        AccuracyHook(dev_feed_dicts, predict, placeholders['targets'], 2)
    ]

    train(loss, optim, train_feed_dicts, max_epochs=args.epochs, hooks=hooks)

    # TODO: evaluate on test data


if __name__ == "__main__":
    main()
