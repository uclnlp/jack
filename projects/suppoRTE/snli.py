# -*- coding: utf-8 -*-

import argparse

import os.path as path
import tensorflow as tf

from time import time

from jtr.preprocess.batch import get_feed_dicts
from jtr.preprocess.vocab import NeuralVocab
from jtr.train import train
from jtr.util.hooks import ExamplesPerSecHook, LossHook, EvalHook
from jtr.pipelines import simple_pipeline, create_placeholders
import jtr.nn.models as models
from jtr.util.rs import DefaultRandomState

from jtr.load.embeddings.embeddings import load_embeddings
from jtr.load.read_jtr import jtr_load

from .kvrte import key_value_rte

tf.set_random_seed(1337)


class Duration(object):
    def __init__(self):
        self.t0 = time()
        self.t = time()

    def __call__(self):
        print('Time since last checkpoint : %.2fmin'%((time()-self.t)/60.))
        self.t = time()

checkpoint = Duration()


def map_to_targets(xs, cands_name, ans_name):
    """
    WARNING: will be moved to jtr.sisyphos.pipelines
    Create cand-length vector for each training instance with 1.0s for cands which are the correct answ and 0.0s for cands which are the wrong answ
    """
    targs = []
    for i in range(len(xs[ans_name])):
        targ = []
        for cand in xs[cands_name][i]:
            if xs[ans_name][i] == cand:
                targ.append(1.0)
            else:
                targ.append(0.0)
        targs.append(targ)
    xs["targets"] = targs
    return xs



def main():

    t0 = time()
    # this is where the list of all models lives, add those if they work
    reader_models = {
        'bicond_singlesupport_reader': models.conditional_reader_model
    }

    support_alts = {'none', 'single', 'multiple'}
    question_alts = answer_alts = {'single', 'multiple'}
    candidate_alts = {'open', 'per-instance', 'fixed'}


    train_default = "./jtr/data/SNLI/snli_1.0/snli_1.0_train_jtr_v1.json"
    dev_default = "./jtr/data/SNLI/snli_1.0/snli_1.0_dev_jtr_v1.json"
    test_default = "./jtr/data/SNLI/snli_1.0/snli_1.0_test_jtr_v1.json"


    #args
    parser = argparse.ArgumentParser(description='Train and Evaluate a machine reader')
    parser.add_argument('--debug', default='False', choices={'True','False'}, help="Run in debug mode, in which case the training file is also used for testing (default False)")
    parser.add_argument('--debug_examples', default=2000, type=int, help="If in debug mode, how many examples should be used (default 2000)")
    parser.add_argument('--train', default=train_default, type=argparse.FileType('r'), help="jtr training file")
    parser.add_argument('--dev', default=dev_default, type=argparse.FileType('r'), help="jtr dev file")
    parser.add_argument('--test', default=test_default, type=argparse.FileType('r'), help="jtr test file")
    parser.add_argument('--supports', default='single', choices=sorted(support_alts), help="None, single (default), or multiple supporting statements per instance")
    parser.add_argument('--questions', default='single', choices=sorted(question_alts), help="None, single (default), or multiple questions per instance")
    parser.add_argument('--candidates', default='fixed', choices=sorted(candidate_alts), help="Open, per-instance, or fixed (default) candidates")
    parser.add_argument('--answers', default='single', choices=sorted(answer_alts), help="Open, per-instance, or fixed (default) candidates")
    parser.add_argument('--batch_size', default=256, type=int, help="Batch size for training data, default 32")
    parser.add_argument('--dev_batch_size', default=256, type=int, help="Batch size for development data, default 32")
    parser.add_argument('--repr_dim_input', default=300, type=int, help="Size of the input representation (embeddings), default 100 (embeddings cut off or extended if not matched with pretrained embeddings)")
    parser.add_argument('--repr_dim_input_reduced', default=100, type=int,
                        help="Size of the input embeddings after reducing with fully_connected layer (default 100)")
    parser.add_argument('--repr_dim_output', default=100, type=int, help="Size of the output representation, default 100")
    parser.add_argument('--pretrain', default='False', choices={'True','False'}, help="Use pretrained embeddings, by default the initialisation is random, default False")
    parser.add_argument('--train_pretrain', default='False', choices={'True','False'},
                        help="Continue training pretrained embeddings together with model parameters, default False")
    parser.add_argument('--normalize_pretrain', default='True', choices={'True','False'},
                        help="Normalize pretrained embeddings, default True (randomly initialized embeddings have expected unit norm too)")
    parser.add_argument('--model', default='bicond_singlesupport_reader_with_cands', choices=sorted(reader_models.keys()), help="Reading model to use")
    parser.add_argument('--learning_rate', default=0.003, type=float, help="Learning rate, default 0.001")
    parser.add_argument('--l2', default=0.0, type=float, help="L2 regularization weight, default 0.0")
    parser.add_argument('--clip_value', default=0.0, type=float, help="gradients clipped between [-clip_value, clip_value] (default 0.0; no clipping)")
    parser.add_argument('--drop_keep_prob', default=1.0, type=float, help="keep probability for dropout on output (set to 1.0 for no dropout)")
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs to train for, default 5")
    parser.add_argument('--tokenize', default='True', choices={'True','False'},help="Tokenize question and support, default True")
    #parser.add_argument('--negsamples', default=0, type=int, help="Number of negative samples, default 0 (= use full candidate list)")
    parser.add_argument('--tensorboard_folder', default='./.tb/', help='Folder for tensorboard logs')
    parser.add_argument('--experimental', default='False', choices={'True','False'}, help="Use experimental SNLI models (default False)")
    parser.add_argument('--buckets', default=5, type=int, help="Number of buckets per field, default 5")
    parser.add_argument('--seed', default=1337, type=int, help='random seed')


    args = parser.parse_args()
    #todo: see if tf.app.flags is more convenient

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
        args.experimental = read_bool(args.experimental)
    _prep_args()


    #set random seed
    tf.set_random_seed(args.seed)
    DefaultRandomState(args.seed)


    #print out args
    print('configuration:')
    for arg in vars(args):
        print('\t%s : %s'%(str(arg), str(getattr(args, arg))))

    if args.debug:
        train_data, dev_data, test_data = [jtr_load(name, args.debug_examples, **vars(args))
                                           for name in [args.train, args.dev, args.test]]
        print('loaded at most %d samples each from debug train/dev/test data'%args.debug_examples)
        if args.pretrain:
            emb_file = 'glove.6B.50d.txt'
            embeddings = load_embeddings(path.join('jtr', 'data', 'GloVe', emb_file), 'glove')
            print('loaded pre-trained embeddings (%s)'%emb_file)
    else:
        train_data, dev_data, test_data = [jtr_load(name,**vars(args)) for name in [args.train, args.dev, args.test]]
        print('loaded train/dev/test data')
        if args.pretrain:
            emb_file = 'GoogleNews-vectors-negative300.bin.gz'
            embeddings = load_embeddings(path.join('jtr', 'data', 'word2vec', emb_file),'word2vec')
            #emb_file = 'glove.840B.300d.zip'
            #embeddings = load_embeddings(path.join('jtr', 'data', 'GloVe', emb_file), 'glove')
            print('loaded pre-trained embeddings (%s)'%emb_file)

    emb = embeddings.get if args.pretrain else None
    

    checkpoint()
    print('encode train data')

    train_data, train_vocab, candidate_vocab = simple_pipeline(train_data, emb=emb)
    train_vocab.freeze()
    candidate_vocab.freeze()
    placeholders = create_placeholders(train_data)

    N_oov = train_vocab.count_oov()
    N_pre = train_vocab.count_pretrained()
    print('In Training data vocabulary: %d pre-trained, %d out-of-vocab.' % (N_pre, N_oov))

    vocab_size = len(train_vocab)
    candidate_size = len(candidate_vocab)

    # @todo: we should allow to set vocab_size for smaller vocab; modify pipeline (with suitable normalization of indices)

    # this is a bit of a hack since args are supposed to be user-defined, but it's cleaner that way with passing on args to reader models
    parser.add_argument('--vocab_size', default=vocab_size, type=int)
    parser.add_argument('--answer_size', default=candidate_size, type=int)
    #todo: make answer_size correct in more general case
    args = parser.parse_args()
    _prep_args()

    print("\tvocab size:  %d" % vocab_size)
    print("\tcandidate size: %d" % candidate_size)
    checkpoint()

    print('encode dev data')
    dev_data,  _, _ = simple_pipeline(dev_data, train_vocab, candidate_vocab)
    checkpoint()
    print('encode test data')
    test_data,  _, _ = simple_pipeline(test_data, train_vocab, candidate_vocab)
    checkpoint()

    print('build NeuralVocab')
    nvocab = NeuralVocab(train_vocab, input_size=args.repr_dim_input, use_pretrained=args.pretrain,
                         reduced_input_size=args.repr_dim_input_reduced,
                         train_pretrained=args.train_pretrain, unit_normalize=args.normalize_pretrain)

    checkpoint()
    print('build model %s'%args.model)

    reader_model = key_value_rte if args.experimental else models.conditional_reader_model

    logits, loss, predict = reader_model(placeholders, nvocab, **vars(args))
    #logits, loss, predict = models.boe_reader_model(placeholders, nvocab, **vars(args))
    #todo: get rid of targets

    if args.supports != "none":
        bucket_order = ('question','support') #composite buckets; first over question, then over support
        bucket_structure = (args.buckets, args.buckets) #args.buckets^2 composite buckets, evenly spaced over questions and supports
    else:
        bucket_order = ('question',) #question buckets
        bucket_structure = (args.buckets,) #args.buckets buckets, evenly spaced over questions

    train_feed_dicts = \
        get_feed_dicts(train_data, placeholders, args.batch_size,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)
    dev_feed_dicts = \
        get_feed_dicts(dev_data, placeholders, args.dev_batch_size,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)

    test_feed_dicts = \
        get_feed_dicts(test_data, placeholders, args.dev_batch_size,
                       bucket_order=bucket_order, bucket_structure=bucket_structure)

    optim = tf.train.AdamOptimizer(args.learning_rate)

    dev_feed_dict = next(dev_feed_dicts.__iter__()) #little bit hacky..; for visualization of dev data during training
    sw = tf.train.SummaryWriter(args.tensorboard_folder)

#    if "cands" in args.model:
#        answname = "targets"
#    else:
#        answname = "answers"
    answname = 'answers'


    hooks = [
        # report_loss,
        LossHook(100, args.batch_size, summary_writer=sw),
        ExamplesPerSecHook(100, args.batch_size, summary_writer=sw),
        #TensorHook(20, [loss, nvocab.get_embedding_matrix()],
        #           feed_dicts=dev_feed_dicts, summary_writer=sw, modes=['min', 'max', 'mean_abs']),
        EvalHook(train_feed_dicts, logits, predict, placeholders[answname],
                 at_every_epoch=1, metrics=['Acc','macroF1'], print_details=False, info="training",
                 summary_writer=sw),
        EvalHook(dev_feed_dicts, logits, predict, placeholders[answname],
                 at_every_epoch=1, metrics=['Acc','macroF1'], print_details=False, info="development",
                 summary_writer=sw),
        EvalHook(test_feed_dicts, logits, predict, placeholders[answname],
                    at_every_epoch=args.epochs, metrics=['Acc','macroP','macroR','macroF1'], print_details=False, info="test data", print_to="")
        # set print_details to True to see gold + pred for all test instances
    ]


    train(loss, optim, train_feed_dicts, max_epochs=args.epochs, l2=args.l2, clip=args.clip_value, hooks=hooks)


    print('finished in %.3fh' % ((time() - t0) / 3600.))


if __name__ == "__main__":
    main()
