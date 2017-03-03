# -*- coding: utf-8 -*-

import argparse
import os
import os.path as path

from time import time
import sys

import logging

import tensorflow as tf

from jtr.preprocess.batch import get_feed_dicts
from jtr.preprocess.vocab import NeuralVocab
from jtr.train import train
from jtr.util.hooks import ExamplesPerSecHook, LossHook, EvalHook
import jtr.nn.models as models
from jtr.load.embeddings.embeddings import load_embeddings
from jtr.pipelines import create_placeholders, pipeline
from jtr.util.rs import DefaultRandomState

from jtr.load.read_jtr import jtr_load
from tensorflow.python.client import device_lib

logger = logging.getLogger(os.path.basename(sys.argv[0]))


class Duration(object):
    def __init__(self):
        self.t0 = time()
        self.t = time()

    def __call__(self):
        logger.info('Time since last checkpoint : {0:.2g}min'.format((time()-self.t)/60.))
        self.t = time()

checkpoint = Duration()

"""
    Loads data, preprocesses it, and finally initializes and trains a model.

   The script does step-by-step:
      (1) Define JTR models
      (2) Parse the input arguments
      (3) Read the train, dev, and test data (with optionally loading pretrained embeddings)
      (4) Preprocesses the data (tokenize, normalize, add start and end of sentence tags) via the sisyphos.pipeline method
      (5) Create NeuralVocab
      (6) Create TensorFlow placeholders and initialize model
      (7) Batch the data via jtr.preprocess.batch.get_feed_dicts
      (8) Add hooks
      (9) Train the model
"""


def main():
    t0 = time()

    # (1) Defined JTR models
    # Please add new models to models.__models__ when they work
    reader_models = {model_name: models.get_function(model_name) for model_name in models.__models__}

    support_alts = {'none', 'single', 'multiple'}
    question_alts = answer_alts = {'single', 'multiple'}
    candidate_alts = {'open', 'per-instance', 'fixed'}

    train_default = dev_default = test_default = '../tests/test_data/sentihood/overfit.json'

    # (2) Parse the input arguments
    parser = argparse.ArgumentParser(description='Train and Evaluate a Machine Reader')

    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode, in which case the training file is also used for testing")
    parser.add_argument('--debug_examples', default=10, type=int,
                        help="If in debug mode, how many examples should be used (default 2000)")
    parser.add_argument('--train', default=train_default, type=argparse.FileType('r'), help="jtr training file")
    parser.add_argument('--dev', default=dev_default, type=argparse.FileType('r'), help="jtr dev file")
    parser.add_argument('--test', default=test_default, type=argparse.FileType('r'), help="jtr test file")
    parser.add_argument('--supports', default='single', choices=sorted(support_alts),
                        help="None, single (default) or multiple supporting statements per instance; "
                             "multiple_flat reads multiple instances creates a separate instance for every support")
    parser.add_argument('--questions', default='single', choices=sorted(question_alts),
                        help="None, single (default), or multiple questions per instance")
    parser.add_argument('--candidates', default='fixed', choices=sorted(candidate_alts),
                        help="Open, per-instance, or fixed (default) candidates")
    parser.add_argument('--answers', default='single', choices=sorted(answer_alts),
                        help="Single or multiple")
    parser.add_argument('--batch_size', default=128,
                        type=int, help="Batch size for training data, default 128")
    parser.add_argument('--dev_batch_size', default=128,
                        type=int, help="Batch size for development data, default 128")
    parser.add_argument('--repr_dim_input', default=300, type=int,
                        help="Size of the input representation (embeddings),"
                             "default 100 (embeddings cut off or extended if not matched with pretrained embeddings)")
    parser.add_argument('--repr_dim_input_trf', default=100, type=int,
                        help="Size of the input embeddings after reducing with fully_connected layer (default 100)")
    parser.add_argument('--repr_dim_output', default=100, type=int,
                        help="Size of the output representation, default 100")

    parser.add_argument('--pretrain', action='store_true',
                        help="Use pretrained embeddings, by default the initialisation is random")
    parser.add_argument('--train_pretrain', action='store_true',
                        help="Continue training pretrained embeddings together with model parameters")
    parser.add_argument('--normalize_pretrain', action='store_true',
                        help="Normalize pretrained embeddings, default False "
                             "(randomly initialized embeddings have expected unit norm too)")

    parser.add_argument('--vocab_maxsize', default=sys.maxsize, type=int)
    parser.add_argument('--vocab_minfreq', default=2, type=int)
    parser.add_argument('--vocab_sep', default=True, type=bool,
                        help='Should there be separate vocabularies for questions and supports, '
                             'vs. candidates and answers. This needs to be set to True for candidate-based methods.')
    parser.add_argument('--model', default='bicond_singlesupport_reader', choices=sorted(reader_models.keys()), help="Reading model to use")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate, default 0.001")
    parser.add_argument('--l2', default=0.0, type=float, help="L2 regularization weight, default 0.0")
    parser.add_argument('--clip_value', default=None, type=float,
                        help="Gradients clipped between [-clip_value, clip_value] (default: no clipping)")
    parser.add_argument('--drop_keep_prob', default=1.0, type=float,
                        help="Keep probability for dropout on output (set to 1.0 for no dropout)")
    parser.add_argument('--epochs', default=5, type=int, help="Number of epochs to train for, default 5")

    parser.add_argument('--tokenize', dest='tokenize', action='store_true', help="Tokenize question and support")
    parser.add_argument('--no-tokenize', dest='tokenize', action='store_false', help="Tokenize question and support")
    parser.set_defaults(tokenize=True)
    parser.add_argument('--lowercase', dest='lowercase', action='store_true', help="Lowercase data")

    parser.add_argument('--negsamples', default=0, type=int,
                        help="Number of negative samples, default 0 (= use full candidate list)")
    parser.add_argument('--tensorboard_folder', default='./.tb/',
                        help='Folder for tensorboard logs')
    parser.add_argument('--write_metrics_to', default=None, type=str,
                        help='Filename to log the metrics of the EvalHooks')
    parser.add_argument('--prune', default='False',
                        help='If the vocabulary should be pruned to the most frequent words.')
    parser.add_argument('--seed', default=1337, type=int, help='random seed')
    parser.add_argument('--logfile', default=None, type=str, help='log file')

    args = parser.parse_args()

    clip_range = (- abs(args.clip_value), abs(args.clip_value)) if args.clip_value else None

    if args.logfile:
        fh = logging.FileHandler(args.logfile)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(levelname)s:%(name)s:\t%(message)s'))
        logger.addHandler(fh)

    logger.info('Configuration:')
    for arg in vars(args):
        logger.info('\t{} : {}'.format(str(arg), str(getattr(args, arg))))

    # set random seed
    tf.set_random_seed(args.seed)
    DefaultRandomState(args.seed)

    # Get information about available CPUs and GPUs:
    # to set specific device, add CUDA_VISIBLE_DEVICES environment variable, e.g.
    # $ CUDA_VISIBLE_DEVICES=0 ./jtr_script.py

    logger.info('available devices:')
    for l in device_lib.list_local_devices():
        logger.info('device info: ' + str(l).replace("\n", " "))

    # (3) Read the train, dev, and test data (with optionally loading pre-trained embeddings
    embeddings = None
    train_data, dev_data, test_data = None, None, None

    if args.debug:
        train_data = jtr_load(args.train, args.debug_examples, **vars(args))
        dev_data, test_data = train_data, train_data

        logger.info('Loaded {} samples as debug train/dev/test dataset '.format(args.debug_examples))

        if args.pretrain:
            emb_file = 'glove.6B.50d.txt'
            embeddings = load_embeddings(path.join('jtr', 'data', 'GloVe', emb_file), 'glove')
            logger.info('loaded pre-trained embeddings ({})'.format(emb_file))
    else:
        if args.train:
            train_data = jtr_load(args.train, **vars(args))

        if args.dev:
            dev_data = jtr_load(args.dev, **vars(args))

        if args.test:
            test_data = jtr_load(args.test, **vars(args))

        logger.info('loaded train/dev/test data')
        if args.pretrain:
            emb_file = 'GoogleNews-vectors-negative300.bin.gz'
            embeddings = load_embeddings(path.join('jtr', 'data', 'SG_GoogleNews', emb_file), 'word2vec')
            logger.info('loaded pre-trained embeddings ({})'.format(emb_file))

    emb = embeddings.get if args.pretrain else None

    checkpoint()

    #  (4) Preprocesses the data (tokenize, normalize, add
    #  start and end of sentence tags) via the JTR pipeline method

    if args.vocab_minfreq != 0 and args.vocab_maxsize != 0:
        logger.info('build vocab based on train data')
        _, train_vocab, train_answer_vocab, train_candidate_vocab = pipeline(train_data, normalize=True,
                                                                             sepvocab=args.vocab_sep,
                                                                             tokenization=args.tokenize,
                                                                             lowercase=args.lowercase,
                                                                             emb=emb)
        if args.prune == 'True':
            train_vocab = train_vocab.prune(args.vocab_minfreq, args.vocab_maxsize)

        logger.info('encode train data')
        train_data, _, _, _ = pipeline(train_data, train_vocab, train_answer_vocab, train_candidate_vocab,
                                       normalize=True, freeze=True, sepvocab=args.vocab_sep,
                                       tokenization=args.tokenize, lowercase=args.lowercase, negsamples=args.negsamples)
    else:
        train_data, train_vocab, train_answer_vocab, train_candidate_vocab = pipeline(train_data, emb=emb,
                                                                                      normalize=True,
                                                                                      tokenization=args.tokenize,
                                                                                      lowercase=args.lowercase,
                                                                                      negsamples=args.negsamples,
                                                                                      sepvocab=args.vocab_sep)

    N_oov = train_vocab.count_oov()
    N_pre = train_vocab.count_pretrained()
    logger.info('In Training data vocabulary: {} pre-trained, {} out-of-vocab.'.format(N_pre, N_oov))

    vocab_size = len(train_vocab)
    answer_size = len(train_answer_vocab)

    # this is a bit of a hack since args are supposed to be user-defined,
    # but it's cleaner that way with passing on args to reader models
    parser.add_argument('--vocab_size', default=vocab_size, type=int)
    parser.add_argument('--answer_size', default=answer_size, type=int)
    args = parser.parse_args()

    checkpoint()
    logger.info('encode dev data')
    dev_data, _, _, _ = pipeline(dev_data, train_vocab, train_answer_vocab, train_candidate_vocab, freeze=True,
                                 tokenization=args.tokenize, lowercase=args.lowercase, sepvocab=args.vocab_sep)
    checkpoint()
    logger.info('encode test data')
    test_data, _, _, _ = pipeline(test_data, train_vocab, train_answer_vocab, train_candidate_vocab, freeze=True,
                                  tokenization=args.tokenize, lowercase=args.lowercase, sepvocab=args.vocab_sep)
    checkpoint()

    # (5) Create NeuralVocab
    logger.info('build NeuralVocab')
    nvocab = NeuralVocab(train_vocab, input_size=args.repr_dim_input, reduced_input_size=args.repr_dim_input_trf,
                         use_pretrained=args.pretrain,
                         train_pretrained=args.train_pretrain, unit_normalize=args.normalize_pretrain)

    with tf.variable_scope("candvocab"):
        candvocab = NeuralVocab(train_candidate_vocab, input_size=args.repr_dim_input,
                                reduced_input_size=args.repr_dim_input_trf, use_pretrained=args.pretrain,
                                train_pretrained=args.train_pretrain, unit_normalize=args.normalize_pretrain)
    checkpoint()

    # (6) Create TensorFlow placeholders and initialize model
    logger.info('create placeholders')
    placeholders = create_placeholders(train_data)
    logger.info('build model {}'.format(args.model))

    # add dropout on the model level
    # todo: more general solution
    options_train = vars(args)
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None):
            (logits_train, loss_train, predict_train) = reader_models[args.model](placeholders,
                                                                                  nvocab,
                                                                                  candvocab=candvocab,
                                                                                  **options_train)

    options_valid = {k: v for k, v in options_train.items()}
    options_valid["drop_keep_prob"] = 1.0
    with tf.name_scope("Valid_Test"):
        with tf.variable_scope("Model", reuse=True):
            (logits_valid, loss_valid, predict_valid) = reader_models[args.model](placeholders,
                                                                                  nvocab,
                                                                                  candvocab=candvocab,
                                                                                  **options_valid)

    # (7) Batch the data via jtr.batch.get_feed_dicts
    if args.supports != "none":
        # composite buckets; first over question, then over support
        bucket_order = ('question', 'support')
        # will result in 16 composite buckets, evenly spaced over questions and supports
        bucket_structure = (1, 1)  # (4, 4)
    else:
        # question buckets
        bucket_order = ('question',)
        # 4 buckets, evenly spaced over questions
        bucket_structure = (1,)  # (4,)

    train_feed_dicts = get_feed_dicts(train_data, placeholders, args.batch_size,
                                      bucket_order=bucket_order, bucket_structure=bucket_structure, exact_epoch=False)
    dev_feed_dicts = get_feed_dicts(dev_data, placeholders, args.dev_batch_size, exact_epoch=True)

    test_feed_dicts = get_feed_dicts(test_data, placeholders, args.dev_batch_size, exact_epoch=True)

    optim = tf.train.AdamOptimizer(args.learning_rate)

    sw = tf.summary.FileWriter(args.tensorboard_folder)

    answname = "targets" if "cands" in args.model else "answers"

    # (8) Add hooks
    hooks = [
        # report_loss
        LossHook(1, args.batch_size, summary_writer=sw),
        ExamplesPerSecHook(100, args.batch_size, summary_writer=sw),

        # evaluate on train data after each epoch
        EvalHook(train_feed_dicts, logits_valid, predict_valid, placeholders[answname],
                 at_every_epoch=1, metrics=['Acc', 'macroF1'],
                 print_details=False, write_metrics_to=args.write_metrics_to, info="training", summary_writer=sw),

        # evaluate on dev data after each epoch
        EvalHook(dev_feed_dicts, logits_valid, predict_valid, placeholders[answname],
                 at_every_epoch=1, metrics=['Acc', 'macroF1'], print_details=False,
                 write_metrics_to=args.write_metrics_to, info="development", summary_writer=sw),

        # evaluate on test data after training
        EvalHook(test_feed_dicts, logits_valid, predict_valid, placeholders[answname],
                 at_every_epoch=args.epochs, metrics=['Acc', 'macroP', 'macroR', 'macroF1'],
                 print_details=False, write_metrics_to=args.write_metrics_to, info="test")
    ]

    # (9) Train the model
    train(loss_train, optim, train_feed_dicts, max_epochs=args.epochs, l2=args.l2, clip=clip_range, hooks=hooks)
    logger.info('finished in {0:.3g}'.format((time() - t0) / 3600.))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
