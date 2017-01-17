import argparse
import json
import os.path as path
import tensorflow as tf
tf.set_random_seed(1337)

from time import time
import sys
from pprint import pprint

class Duration(object):
    def __init__(self):
        self.t0 = time()
        self.t = time()
    def __call__(self):
        print('Time since last checkpoint : %.2fmin'%((time()-self.t)/60.))
        self.t = time()

checkpoint = Duration()


from quebap.sisyphos.batch import get_feed_dicts
from quebap.sisyphos.vocab import Vocab, NeuralVocab
from quebap.sisyphos.map import tokenize, lower, deep_map, deep_seq_map, dynamic_subsample
from quebap.sisyphos.train import train
from quebap.sisyphos.hooks import ExamplesPerSecHook, AccuracyHook, LossHook, TensorHook, EvalHook
import quebap.model.models as models
from quebap.load.embeddings.embeddings import load_embeddings
from quebap.sisyphos.pipelines import create_placeholders, pipeline

from quebap.load.read_quebap import quebap_load as _quebap_load

"""Loads data, preprocesses it, and finally initializes and trains a model.

   The script does step-by-step:
      (1) Define sisyphos models
      (2) Parse the input arguments
      (3) Read the train, dev, and test data
          (with optionally loading pretrained embeddings)
      (4) Preprocesses the data (tokenize, normalize, add
          start and end of sentence tags) via the sisyphos.pipeline method
      (5) Create NeuralVocab
      (6) Create TensorFlow placeholders and initialize model
      (7) Batch the data via sisyphos.batch.get_feed_dicts
      (8) Add hooks
      (9) Train the model
"""

def quebap_load(path, max_count=None, **options):
    return _quebap_load(path, max_count, **options)

def main():

    t0 = time()

    #(1) Defined sisyphos models

    # this is where the list of all models lives, add those if they work
    reader_models = {
        'bicond_singlesupport_reader': models.conditional_reader_model,
        'bicond_singlesupport_reader_with_cands': models.conditional_reader_model_with_cands,
        'bilstm_singlesupport_reader_with_cands': models.bilstm_reader_model_with_cands,
        'bilstm_nosupport_reader_with_cands': models.bilstm_nosupport_reader_model_with_cands,
        'boe_support_cands': models.boe_support_cands_reader_model,
        'boe_nosupport_cands': models.boe_nosupport_cands_reader_model,
        'boe_support': models.boe_reader_model,
        'boe_nosupport': models.boenosupport_reader_model,
        #'log_linear': ReaderModel.create_log_linear_reader,
        #'model_f': ReaderModel.create_model_f_reader,
    }

    support_alts = {'none', 'single', 'multiple'}
    question_alts = answer_alts = {'single', 'multiple'}
    candidate_alts = {'open', 'per-instance', 'fixed'}

    #todo clean up
    #common default input files - for rapid testing
    #train_default = 'data/SQuAD/snippet_quebapformat.json'
    #dev_default = 'data/sentihood/single_quebap.json'
    #test_default = 'data/sentihood/single_quebap.json'
    #train_default = "./quebap/data/SNLI/snli_1.0/snli_1.0_train_quebap_v1.json"
    #dev_default = "./quebap/data/SNLI/snli_1.0/snli_1.0_dev_quebap_v1.json"
    #test_default = "./quebap/data/SNLI/snli_1.0/snli_1.0_test_quebap_v1.json"

    train_default = dev_default = test_default = 'data/SNLI/snippet_quebapformat_v1.json'
    #train_default = dev_default = test_default = 'data/scienceQA/scienceQA_cloze_snippet.json'
    #train_default = dev_default = test_default = '../tests/test_data/sentihood/overfit.json'

    #(2) Parse the input arguments

    parser = argparse.ArgumentParser(description='Train and Evaluate a machine reader')
    parser.add_argument('--debug', default='False', choices={'True','False'}, help="Run in debug mode, in which case the training file is also used for testing (default False)")
    parser.add_argument('--debug_examples', default=10, type=int, help="If in debug mode, how many examples should be used (default 2000)")
    parser.add_argument('--train', default=train_default, type=argparse.FileType('r'), help="Quebap training file")
    parser.add_argument('--dev', default=dev_default, type=argparse.FileType('r'), help="Quebap dev file")
    parser.add_argument('--test', default=test_default, type=argparse.FileType('r'), help="Quebap test file")
    parser.add_argument('--supports', default='single', choices=sorted(support_alts), help="None, single (default) or multiple supporting statements per instance; multiple_flat reads multiple instances creates a separate instance for every support")
    parser.add_argument('--questions', default='single', choices=sorted(question_alts), help="None, single (default), or multiple questions per instance")
    parser.add_argument('--candidates', default='fixed', choices=sorted(candidate_alts), help="Open, per-instance, or fixed (default) candidates")
    parser.add_argument('--answers', default='single', choices=sorted(answer_alts), help="Open, per-instance, or fixed (default) candidates")
    parser.add_argument('--batch_size', default=128,
        type=int, help="Batch size for training data, default 128")
    parser.add_argument('--dev_batch_size', default=128,
        type=int, help="Batch size for development data, default 128")
    parser.add_argument('--repr_dim_input', default=100, type=int, help="Size of the input representation (embeddings), default 100 (embeddings cut off or extended if not matched with pretrained embeddings)")
    parser.add_argument('--repr_dim_output', default=100, type=int, help="Size of the output representation, default 100")
    parser.add_argument('--pretrain', default='False', choices={'True','False'}, help="Use pretrained embeddings, by default the initialisation is random, default False")
    parser.add_argument('--train_pretrain', default='False', choices={'True','False'},
                        help="Continue training pretrained embeddings together with model parameters, default False")
    parser.add_argument('--normalize_pretrain', default='True', choices={'True','False'},
                        help="Normalize pretrained embeddings, default True (randomly initialized embeddings have expected unit norm too)")
    parser.add_argument('--vocab_maxsize', default=sys.maxsize, type=int)
    parser.add_argument('--vocab_minfreq', default=2, type=int)
    parser.add_argument('--model', default='bicond_singlesupport_reader', choices=sorted(reader_models.keys()), help="Reading model to use")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate, default 0.001")
    parser.add_argument('--l2', default=0.0, type=float, help="L2 regularization weight, default 0.0")
    parser.add_argument('--clip_value', default=0.0, type=float, help="gradients clipped between [-clip_value, clip_value] (default 0.0; no clipping)")
    parser.add_argument('--drop_keep_prob', default=0.9, type=float, help="keep probability for dropout on output (set to 1.0 for no dropout)")
    parser.add_argument('--epochs', default=5, type=int, help="Number of epochs to train for, default 5")
    parser.add_argument('--tokenize', default='True', choices={'True','False'},help="Tokenize question and support, default True")
    parser.add_argument('--negsamples', default=0, type=int, help="Number of negative samples, default 0 (= use full candidate list)")
    parser.add_argument('--tensorboard_folder', default='./.tb/', help='Folder for tensorboard logs')
    parser.add_argument('--write_metrics_to', default='',
        help='Filename to log the metrics of the EvalHooks')
    #parser.add_argument('--train_begin', default=0, metavar='B', type=int, help="Use if training and test are the same file and the training set needs to be split. Index of first training instance.")
    #parser.add_argument('--train_end', default=-1, metavar='E', type=int,
    #                    help="Use if training and test are the same file and the training set needs to be split. Index of last training instance plus 1.")
    #parser.add_argument('--candidate_split', default="$", type=str, metavar="S",
    #                    help="Regular Expression for tokenizing candidates. By default candidates are not split")
    #parser.add_argument('--question_split', default="-", type=str, metavar="S",
    #                    help="Regular Expression for tokenizing questions")
    #parser.add_argument('--support_split', default="-", type=str, metavar="S",
    #                    help="Regular Expression for tokenizing support")
    #parser.add_argument('--use_train_generator_for_test', default=False, type=bool, metavar="B",
    #                    help="Should the training candidate generator be used when testing")
    #parser.add_argument('--feature_type', default=None, type=str, metavar="F",
    #                    help="When using features: type of features.")

    args = parser.parse_args()

    #pre-process arguments
    #(hack to circumvent lack of 'bool' type in parser)
    def _prep_args():
        read_bool = lambda l: {'True': True, 'False': False}[l]
        args.debug = read_bool(args.debug)
        args.pretrain = read_bool(args.pretrain)
        args.train_pretrain = read_bool(args.train_pretrain)
        args.normalize_pretrain = read_bool(args.normalize_pretrain)
        args.tokenize = read_bool(args.tokenize)
        args.clip_value = None if args.clip_value == 0.0 else (-abs(args.clip_value),abs(args.clip_value))
    _prep_args()

    #print out args
    print('configuration:')
    for arg in vars(args):
        print('\t%s : %s'%(str(arg),str(getattr(args, arg))))

    #(3) Read the train, dev, and test data
    #    (with optionally loading pretrained embeddings)

    if args.debug:
        train_data = quebap_load(args.train, args.debug_examples, **vars(args))
        print('loaded %d samples as debug train/dev/test dataset '%args.debug_examples)
        dev_data = train_data
        test_data = train_data
        if args.pretrain:
            emb_file = 'glove.6B.50d.txt'
            embeddings = load_embeddings(path.join('quebap', 'data', 'GloVe', emb_file), 'glove')
            print('loaded pre-trained embeddings (%s)'%emb_file)
    else:
        train_data, dev_data, test_data = [quebap_load(name,**vars(args)) for name in [args.train, args.dev, args.test]]
        print('loaded train/dev/test data')
        if args.pretrain:
            emb_file = 'GoogleNews-vectors-negative300.bin.gz'
            embeddings = load_embeddings(path.join('quebap', 'data', 'word2vec', emb_file),'word2vec')
            #emb_file = 'glove.840B.300d.zip'
            #embeddings = load_embeddings(path.join('quebap', 'data', 'GloVe', emb_file), 'glove')
            print('loaded pre-trained embeddings (%s)'%emb_file)

    emb = embeddings.get if args.pretrain else None

    checkpoint()

    #  (4) Preprocesses the data (tokenize, normalize, add
    #      start and end of sentence tags) via the sisyphos.pipeline method

    if args.vocab_minfreq != 0 and args.vocab_maxsize != 0:
        print('build vocab based on train data')
        _, train_vocab, train_answer_vocab, train_candidate_vocab = pipeline(train_data, normalize=True)
        train_vocab = train_vocab.prune(args.vocab_minfreq, args.vocab_maxsize)

        print('encode train data')
        train_data, _, _, _ = pipeline(train_data, train_vocab, train_answer_vocab, train_candidate_vocab, normalize=True, freeze=True)
    else:
        train_data, train_vocab, train_answer_vocab, train_candidate_vocab = pipeline(train_data, emb=emb, normalize=True, tokenization=args.tokenize, negsamples=args.negsamples)

    N_oov = train_vocab.count_oov()
    N_pre = train_vocab.count_pretrained()
    print('In Training data vocabulary: %d pre-trained, %d out-of-vocab.' % (N_pre, N_oov))

    vocab_size = len(train_vocab)
    answer_size = len(train_answer_vocab)

    # this is a bit of a hack since args are supposed to be user-defined, but it's cleaner that way with passing on args to reader models
    parser.add_argument('--vocab_size', default=vocab_size, type=int)
    parser.add_argument('--answer_size', default=answer_size, type=int)
    args = parser.parse_args()
    _prep_args()

    checkpoint()
    print('encode dev data')
    dev_data, _, _, _ = pipeline(dev_data, train_vocab, train_answer_vocab, train_candidate_vocab, freeze=True, tokenization=args.tokenize)
    checkpoint()
    print('encode test data')
    test_data, _, _, _ = pipeline(test_data, train_vocab, train_answer_vocab, train_candidate_vocab, freeze=True, tokenization=args.tokenize)
    checkpoint()

    #(5) Create NeuralVocab

    print('build NeuralVocab')
    nvocab = NeuralVocab(train_vocab, input_size=args.repr_dim_input, use_pretrained=args.pretrain,
                         train_pretrained=args.train_pretrain, unit_normalize=args.normalize_pretrain)
    checkpoint()

    #(6) Create TensorFlow placeholders and intialize model

    print('create placeholders')
    placeholders = create_placeholders(train_data)
    print('build model %s'%args.model)
    (logits, loss, predict) = reader_models[args.model](placeholders, nvocab, **vars(args))

    #(7) Batch the data via sisyphos.batch.get_feed_dicts

    if args.supports != "none":
        bucket_order = ('question','support') #composite buckets; first over question, then over support
        bucket_structure = (4,4) #will result in 16 composite buckets, evenly spaced over questions and supports
    else:
        bucket_order = ('question',) #question buckets
        bucket_structure = (4,) #4 buckets, evenly spaced over questions

    train_feed_dicts = \
        get_feed_dicts(train_data, placeholders, args.batch_size,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)
    dev_feed_dicts = \
        get_feed_dicts(dev_data, placeholders, args.dev_batch_size,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)

    test_feed_dicts = \
        get_feed_dicts(test_data, placeholders, 1,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)

    optim = tf.train.AdamOptimizer(args.learning_rate)

    dev_feed_dict = next(dev_feed_dicts.__iter__()) #little bit hacky..; for visualization of dev data during training
    sw = tf.train.SummaryWriter(args.tensorboard_folder)

    if "cands" in args.model:
        answname = "targets"
    else:
        answname = "answers"

    #(8) Add hooks

    hooks = [
        TensorHook(20, [loss, nvocab.get_embedding_matrix()],
                   feed_dicts=dev_feed_dicts, summary_writer=sw, modes=['min', 'max', 'mean_abs']),
        # report_loss
        LossHook(100, args.batch_size, summary_writer=sw),
        ExamplesPerSecHook(100, args.batch_size, summary_writer=sw),
        #evaluate on train data after each epoch
        EvalHook(train_feed_dicts, logits, predict, placeholders[answname],
                 at_every_epoch=1, metrics=['Acc','macroF1'], print_details=False, write_metrics_to=args.write_metrics_to, info="training",
                 summary_writer=sw),
        #evaluate on dev data after each epoch
        EvalHook(dev_feed_dicts, logits, predict, placeholders[answname],
                 at_every_epoch=1, metrics=['Acc','macroF1'], print_details=False, write_metrics_to=args.write_metrics_to, info="development",
                 summary_writer=sw),
        #evaluate on test data after training
        EvalHook(test_feed_dicts, logits, predict, placeholders[answname],
                    at_every_epoch=args.epochs,
                    metrics=['Acc','macroP','macroR','macroF1'],
                    print_details=False, write_metrics_to=args.write_metrics_to, info="test")
    ]

    #(9) Train the model

    train(loss, optim, train_feed_dicts, max_epochs=args.epochs, l2=args.l2, clip=args.clip_value, hooks=hooks)


    print('finished in %.3fh' % ((time() - t0) / 3600.))


if __name__ == "__main__":
    main()
