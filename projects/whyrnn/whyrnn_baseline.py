# general
import argparse
import sys
import os
import tensorflow as tf
import numpy as np
from time import time
import logging

# jack
import jtr.jack.readers as readers
from jtr.load.embeddings.embeddings import load_embeddings
from jtr.jack.data_structures import load_labelled_data
from jtr.preprocess.vocabulary import Vocab
from jtr.jack.tasks.mcqa.simple_mcqa import SingleSupportFixedClassInputs

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main():
    # input
    train_file = "data/SNLI/snli_1.0/snli_1.0_train_jtr_v1.json"
    dev_file = "data/SNLI/snli_1.0/snli_1.0_dev_jtr_v1.json"
    test_file = "data/SNLI/snli_1.0/snli_1.0_test_jtr_v1.json"

    parser = argparse.ArgumentParser(
        description='Baseline SNLI model experiments')

    # data files
    parser.add_argument('--jtr_path', default='.', help='path to jtr base')

    # debug mode
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode")
    parser.add_argument('--debug_examples', default=2000, type=int,
                        help="If in debug mode, "
                        "how many examples should be used (default 2000)")

    # tensorboard path
    parser.add_argument('--tensorboard_path', default='./.tb/',
                        help='Folder for tensorboard logs')

    # config for preprocessing
    parser.add_argument('--lowercase', action='store_true',
                        help="Lowercase data")

    # config of Vocab
    parser.add_argument('--vocab_max_size', default=sys.maxsize, type=int)
    parser.add_argument('--vocab_min_freq', default=1, type=int)

    # config of embeddings
    parser.add_argument('--pretrain', action='store_true',
                        help="Use pretrained embeddings, "
                        "by default the initialisation is random")
    parser.add_argument('--normalize_embeddings', action='store_true',
                        help="Normalize (initial) embeddings")
    parser.add_argument('--init_embeddings', default='uniform',
                        choices=['uniform', 'normal'])

    # config of model architecture
    parser.add_argument('--hidden_dim', default=100, type=int,
                        help="Size of the hidden representations, default 100")

    # training
    parser.add_argument('--batch_size', default=256,
                        type=int, help="Batch size for training data, "
                        "default 256")
    parser.add_argument('--eval_batch_size', default=256,
                        type=int, help="Batch size when eval=True, "
                        "default 256")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help="Learning rate, default 0.001")
    parser.add_argument('--l2', default=0.0, type=float,
                        help="L2 regularization weight, default 0.0")
    parser.add_argument('--clip_value', default=0, type=float,
                        help="Gradients clipped between "
                        "[-clip_value, clip_value] (default = 0, no clipping)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout probability "
                        "(default 0.0 for no dropout)")
    parser.add_argument('--epochs', default=30, type=int,
                        help="Number of train epochs, default 30")

    # misc
    parser.add_argument('--seed', default=1337, type=int, help='random seed')
    parser.add_argument('--write_metrics_to', default=None, type=str,
                        help='Filename to log the metrics of the EvalHooks')

    args = parser.parse_args()

    # read out explicitly
    jtr_path = args.jtr_path
    debug, debug_examples = args.debug, args.debug_examples
    tensorboard_path = args.tensorboard_path
    lowercase = args.lowercase
    vocab_max_size, vocab_min_freq = args.vocab_max_size, args.vocab_min_freq
    pretrain = args.pretrain
    init_embeddings = args.init_embeddings
    normalize_embeddings = args.normalize_embeddings
    repr_dim_input = 50 if debug else 300
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    learning_rate = args.learning_rate
    dropout, l2, clip_value = args.dropout, args.l2, args.clip_value
    epochs = args.epochs
    write_metrics_to = args.write_metrics_to

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # config params needed for JTReader
    config = {
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'learning_rate': learning_rate,
        'vocab_min_freq': vocab_min_freq,
        'vocab_max_size': vocab_max_size,
        'lowercase': lowercase,
        'repr_dim_input': repr_dim_input,
        'repr_dim': hidden_dim,
        'dropout': dropout,
        'init_embeddings': init_embeddings,
        'normalize_embeddings': normalize_embeddings,
    }

    # logging
    sw = tf.summary.FileWriter(tensorboard_path)

    # load SNLI data
    splits = [train_file, dev_file, test_file]
    max_count = debug_examples if debug else None
    train_set, dev_set, test_set = [
        load_labelled_data(os.path.join(args.jtr_path, f), max_count)
        for f in splits
    ]
    for s, l in zip([train_set, dev_set, test_set], ['train', 'dev', 'test']):
        logger.info('loaded {:d} {:s} instances'.format(len(s), l))

    # load pre-trained embeddings
    embeddings = None
    if pretrain:
        if debug:
            emb_file = 'glove.6B.50d.txt'
            embeddings = load_embeddings(
                os.path.join(jtr_path, 'data', 'GloVe', emb_file),
                'glove')
        else:
            emb_file = 'GoogleNews-vectors-negative300.bin.gz'
            embeddings = load_embeddings(
                os.path.join(jtr_path, 'data', 'SG_GoogleNews', emb_file),
                'word2vec')
        logger.info('loaded pre-trained embeddings ({})'.format(emb_file))

    # create Vocab object
    vocab = Vocab(emb=embeddings)

    # filter dev and test tokens which have pre-trained embeddings
    # (to avoid having to load all)
    if pretrain:
        dev_tmp = SingleSupportFixedClassInputs.preprocess(
            dev_set, lowercase=config['lowercase'],
            test_time=False, add_lengths=False)
        test_tmp = SingleSupportFixedClassInputs.preprocess(
            test_set, lowercase=config['lowercase'],
            test_time=False, add_lengths=False)
        vocab.add_pretrained_for_testing(
            dev_tmp['question'], dev_tmp['support'])
        vocab.add_pretrained_for_testing(
            test_tmp['question'], test_tmp['support'])
        logger.debug(
            'loaded {:d} filtered pretrained symbols into '
            'vocab for dev and test data'.format(len(vocab.symset_pt)))

    # create reader
    reader = readers.readers['snli_reader'](vocab, config)

    # add hooks
    from jtr.jack.train.hooks import LossHook
    hooks = [
        LossHook(reader, iter_interval=50, summary_writer=sw),
        readers.eval_hooks['snli_reader'](
            reader, dev_set, iter_interval=100, info='dev',
            summary_writer=sw, write_metrics_to=write_metrics_to),
        readers.eval_hooks['snli_reader'](
            reader, test_set, epoch_interval=args.epochs,
            info='test', write_metrics_to=write_metrics_to)
    ]
    if args.debug:
        hooks.append(readers.eval_hooks['snli_reader'](
            reader, train_set, iter_interval=100, info='train',
            summary_writer=sw, write_metrics_to=write_metrics_to))

    # Here we initialize our optimizer
    # we choose Adam with standard momentum values
    optim = tf.train.AdamOptimizer(config['learning_rate'])

    t0 = time()
    reader.train(
        optim, train_set,
        hooks=hooks,
        max_epochs=epochs,
        l2=l2,
        clip=None if abs(clip_value) < 1.e-12 else [-clip_value, clip_value]
    )
    # TODO: check device setup in JTReader.train
    print('training took {:.3f} hours'.format((time() - t0) / 3600.))


if __name__ == "__main__":
    main()
