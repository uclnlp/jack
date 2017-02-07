# -*- coding: utf-8 -*-

import argparse
import logging
import math
import os
import os.path as path
import random
import shutil
import sys
from time import time

import tensorflow as tf
from tensorflow.python.client import device_lib

import jtr.jack.readers as readers
from jtr.jack import load_labelled_data
from jtr.jack.train.hooks import LossHook, ExamplesPerSecHook, ETAHook
from jtr.load.embeddings.embeddings import load_embeddings
from jtr.preprocess.vocab import Vocab

logger = logging.getLogger(os.path.basename(sys.argv[0]))


class Duration(object):
    def __init__(self):
        self.t0 = time()
        self.t = time()

    def __call__(self):
        logger.info('Time since last checkpoint : {0:.2g}min'.format((time() - self.t) / 60.))
        self.t = time()


tf.set_random_seed(1337)
checkpoint = Duration()


def main():
    # Please add new models to readers when they work

    support_alts = {'none', 'single', 'multiple'}
    question_alts = answer_alts = {'single', 'multiple'}
    candidate_alts = {'open', 'per-instance', 'fixed'}

    train_default = dev_default = test_default = 'tests/test_data/sentihood/overfit.json'

    parser = argparse.ArgumentParser(description='Train and Evaluate a Machine Reader')
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode, in which case the training file is also used for testing")

    parser.add_argument('--debug_examples', default=10, type=int,
                        help="If in debug mode, how many examples should be used (default 2000)")
    parser.add_argument('--train', default=train_default, type=str, help="jtr training file")
    parser.add_argument('--dev', default=dev_default, type=str, help="jtr dev file")
    parser.add_argument('--test', default='', type=str, help="jtr test file")
    parser.add_argument('--supports', default='single', choices=sorted(support_alts),
                        help="None, single (default) or multiple supporting statements per instance; multiple_flat reads multiple instances creates a separate instance for every support")
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
    parser.add_argument('--repr_dim_input', default=100, type=int,
                        help="Size of the input representation (embeddings), default 100 (embeddings cut off or extended if not matched with pretrained embeddings)")
    parser.add_argument('--repr_dim', default=100, type=int,
                        help="Size of the hidden representations, default 100")

    parser.add_argument('--pretrain', action='store_true',
                        help="Use pretrained embeddings, by default the initialisation is random")
    parser.add_argument('--with_char_embeddings', action='store_true',
                        help="Use also character based embeddings in readers which support it.")
    parser.add_argument('--vocab_from_embeddings', action='store_true',
                        help="Use fixed vocab of pretrained embeddings")
    parser.add_argument('--train_pretrain', action='store_true',
                        help="Continue training pretrained embeddings together with model parameters")
    parser.add_argument('--normalize_pretrain', action='store_true',
                        help="Normalize pretrained embeddings, default True (randomly initialized embeddings have expected unit norm too)")

    parser.add_argument('--embedding_format', default='word2vec', choices=["glove", "word2vec"],
                        help="format of embeddings to be loaded")
    parser.add_argument('--embedding_file', default='jtr/data/SG_GoogleNews/GoogleNews-vectors-negative300.bin.gz',
                        type=str, help="format of embeddings to be loaded")

    parser.add_argument('--vocab_maxsize', default=sys.maxsize, type=int)
    parser.add_argument('--vocab_minfreq', default=2, type=int)
    parser.add_argument('--vocab_sep', default=True, type=bool,
                        help='Should there be separate vocabularies for questions, supports, candidates and answers. This needs to be set to True for candidate-based methods.')
    parser.add_argument('--model', default='fastqa_reader', choices=sorted(readers.readers.keys()),
                        help="Reading model to use")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate, default 0.001")
    parser.add_argument('--learning_rate_decay', default=0.5, type=float, help="Learning rate decay, default 0.5")
    parser.add_argument('--l2', default=0.0, type=float, help="L2 regularization weight, default 0.0")
    parser.add_argument('--clip_value', default=0.0, type=float,
                        help="Gradients clipped between [-clip_value, clip_value] (default 0.0; no clipping)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Probability for dropout on output (set to 0.0 for no dropout)")
    parser.add_argument('--epochs', default=5, type=int, help="Number of epochs to train for, default 5")
    parser.add_argument('--checkpoint', default=None, type=int, help="Number of batches before evaluation on devset.")

    parser.add_argument('--negsamples', default=0, type=int,
                        help="Number of negative samples, default 0 (= use full candidate list)")
    parser.add_argument('--tensorboard_folder', default=None, help='Folder for tensorboard logs')
    parser.add_argument('--write_metrics_to', default='',
                        help='Filename to log the metrics of the EvalHooks')
    parser.add_argument('--prune', default='False',
                        help='If the vocabulary should be pruned to the most frequent words.')
    parser.add_argument('--model_dir', default=None, type=str, help="Directory to write reader to.")
    parser.add_argument('--log_interval', default=100, type=int, help="interval for logging eta, training loss, etc.")
    parser.add_argument('--device', default='/cpu:0', type=str, help='device setting for tensorflow')
    parser.add_argument('--lowercase', action='store_true', help='lowercase texts.')
    parser.add_argument('--seed', default=63783, type=int, help="Seed for rngs.")

    args = parser.parse_args()

    # make everything deterministic
    random.seed(args.seed)
    tf.set_random_seed(args.seed)

    clip_value = None
    if args.clip_value != 0.0:
        clip_value = - abs(args.clip_value), abs(args.clip_value)

    logger.info('configuration:')
    for arg in vars(args):
        logger.info('\t{} : {}'.format(str(arg), str(getattr(args, arg))))

    # Get information about available CPUs and GPUs:
    # to set specific device, add CUDA_VISIBLE_DEVICES environment variable, e.g.
    # $ CUDA_VISIBLE_DEVICES=0 ./jtr_script.py

    logger.info('available devices:')
    for l in device_lib.list_local_devices():
        logger.info('device info: ' + str(l).replace("\n", " "))

    embeddings = None
    if args.debug:
        train_data = load_labelled_data(args.train, args.debug_examples, **vars(args))

        logger.info('loaded {} samples as debug train/dev/test dataset '.format(args.debug_examples))

        dev_data = train_data
        test_data = train_data
        if args.pretrain:
            emb_file = 'glove.6B.50d.txt'
            embeddings = load_embeddings(path.join('jtr', 'data', 'GloVe', emb_file), 'glove')
            logger.info('loaded pre-trained embeddings ({})'.format(emb_file))
            args.repr_input_dim = embeddings.lookup.shape[1]
    else:
        train_data, dev_data = [load_labelled_data(name, **vars(args)) for name in [args.train, args.dev]]
        test_data = load_labelled_data(args.test, **vars(args)) if args.test else None
        logger.info('loaded train/dev/test data')
        if args.pretrain:
            embeddings = load_embeddings(args.embedding_file, args.embedding_format)
            logger.info('loaded pre-trained embeddings ({})'.format(args.embedding_file))
            args.repr_input_dim = embeddings.lookup.shape[1]

    emb = embeddings if args.pretrain else None

    vocab = Vocab(emb=emb, init_from_embeddings=args.vocab_from_embeddings)

    with tf.device(args.device):
        # build JTReader
        checkpoint()
        reader = readers.readers[args.model](vocab, vars(args))
        checkpoint()

        learning_rate = tf.get_variable("learning_rate", initializer=args.learning_rate, dtype=tf.float32,
                                        trainable=False)
        lr_decay_op = learning_rate.assign(args.learning_rate_decay * learning_rate)
        optim = tf.train.AdamOptimizer(learning_rate)

        if args.tensorboard_folder is not None:
            if os.path.exists(args.tensorboard_folder):
                shutil.rmtree(args.tensorboard_folder)
            sw = tf.summary.FileWriter(args.tensorboard_folder)
        else:
            sw = None

        # Hooks
        iter_iterval = 1 if args.debug else args.log_interval
        hooks = [LossHook(reader, iter_iterval, summary_writer=sw),
                 ExamplesPerSecHook(reader, args.batch_size, iter_iterval, sw),
                 ETAHook(reader, iter_iterval, math.ceil(len(train_data) / args.batch_size), args.epochs,
                         args.checkpoint,
                         sw)]

        preferred_metric = "f1"  # TODO: this should depend on the task, for now I set it to 1
        best_metric = [0.0]

        def side_effect(metrics, prev_metric):
            """Returns: a state (in this case a metric) that is used as input for the next call"""
            m = metrics[preferred_metric]
            if prev_metric is not None and m < prev_metric:
                reader.sess.run(lr_decay_op)
                logger.info("Decayed learning rate to: %.5f" % reader.sess.run(learning_rate))
            else:
                best_metric[0] = m
                if prev_metric is None:  # store whole model only at beginning of training
                    reader.store(args.model_dir)
                else:
                    reader.model_module.store(reader.sess, os.path.join(args.model_dir, "model_module"))
                logger.info("Saving model to: %s" % args.model_dir)
            return m

        hooks.append(readers.eval_hooks[args.model](reader, dev_data, summary_writer=sw, side_effect=side_effect,
                                                    iter_interval=args.checkpoint,
                                                    epoch_interval=1 if args.checkpoint is None else None))

        # Train
        reader.train(optim, training_set=train_data,
                     max_epochs=args.epochs, hooks=hooks,
                     l2=args.l2, clip=clip_value, clip_op=tf.clip_by_value,
                     device=args.device)

        # Test final model
        if test_data is not None:
            logger.info(
                "Run evaluation on test set with best model on dev set: %s %.3f" % (preferred_metric, best_metric[0]))
            test_eval_hook = readers.eval_hooks[args.model](reader, test_data, summary_writer=sw, epoch_interval=1)

            reader.load(args.model_dir)
            test_eval_hook(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
