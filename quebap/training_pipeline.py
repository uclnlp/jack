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
from quebap.sisyphos.vocab import Vocab, NeuralVocab
from quebap.sisyphos.map import tokenize, lower, deep_map, deep_seq_map
from quebap.sisyphos.train import train
from quebap.sisyphos.hooks import SpeedHook, AccuracyHook, LossHook
from quebap.model.models import ReaderModel
from quebap.io.embeddings.embeddings import load_embeddings


def quebap_load(path, max_count=None, **options):
    """
    General-purpose loader for quebap files
    Makes use of user-defined options for supports, questions, candidates, answers and only read in those
    things needed for model, e.g. if the dataset contains support, but the user defines support_alts == 'none'
    because they want to train a model that does not make use of support, support information in dataset is not read in

    User options for quebap model/dataset attributes are:
    support_alts = {'none', 'single', 'multiple'}
    question_alts = answer_alts = {'single', 'multiple'}
    candidate_alts = {'open', 'per-instance', 'fixed'}
    """

    reading_dataset = json.load(path)

    def textOrDict(c):
        if isinstance(c, dict):
            c = c["text"]
        return c

    # The script reads into those lists. If IDs for questions, supports or targets are defined, those are ignored.
    questions = []
    supports = []
    answers = []
    candidates = []
    global_candidates = []
    count = 0
    if "globals" in reading_dataset:
        global_candidates = [textOrDict(c) for c in reading_dataset['globals']['candidates']]

    for instance in reading_dataset['instances']:
        question, support, answer, candidate = "", "", "", ""  # initialisation
        if max_count is None or count < max_count:
            if options["supports"] == "single":
                support = textOrDict(instance['support'][0])
            elif options["supports"] == "multiple":
                support = [textOrDict(c) for c in instance['support'][0]]
            if options["questions"] == "single":
                question = textOrDict(instance['questions'][0]["question"]) # if single, just take the first one, could also change this to random
                if options["answers"] == "single":
                    answer = textOrDict(instance['questions'][0]['answers'][0]) # if single, just take the first one, could also change this to random
                elif options["answers"] == "multiple":
                    answer = [textOrDict(c) for c in instance['questions'][0]['answers']]
                if options["candidates"] == "per-instance":
                    candidate = [textOrDict(c) for c in instance['candidates']]

            elif options["questions"] == "multiple":
                answer = []
                candidate = []
                question = [textOrDict(c["question"]) for c in instance['questions']]
                if options["answers"] == "single":
                    answer = [textOrDict(c["answers"][0]) for c in instance['questions']]
                elif options["answers"] == "multiple":
                    answer = [textOrDict(c) for q in instance['questions'] for c in q["answers"]]
                if options["candidates"] == "per-instance":
                    candidate = [textOrDict(c) for quest in instance["questions"] for c in quest["candidates"]]

            if options["candidates"] == "fixed":
                candidates.append(global_candidates)

            questions.append(question)
            answers.append(answer)
            if options["supports"] != "none":
                supports.append(support)
            if options["candidates"] != "fixed":
                candidates.append(candidate)
            count += 1


    print("Loaded %d examples from %s" % (len(questions), path))
    return {'question': questions, 'support': supports, 'answers': answers, 'candidates': candidates}


#@todo: rewrite such that it works for different types of quebap files / models
def pipeline(corpus, vocab=None, target_vocab=None, candidate_vocab=None, emb=None, freeze=False):
    vocab = vocab or Vocab(emb=emb)
    target_vocab = target_vocab or Vocab(unk=None)
    candidate_vocab = candidate_vocab or Vocab(unk=None)
    if freeze:
        vocab.freeze()
        target_vocab.freeze()
        candidate_vocab.freeze()

    corpus_tokenized = deep_map(corpus, tokenize, ['question', 'support'])
    corpus_lower = deep_seq_map(corpus_tokenized, lower, ['question', 'support'])
    corpus_os = deep_seq_map(corpus_lower, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ['question', 'support'])
    corpus_ids = deep_map(corpus_os, vocab, ['question', 'support'])
    corpus_ids = deep_map(corpus_ids, target_vocab, ['answers'])
    corpus_ids = deep_map(corpus_ids, candidate_vocab, ['candidates'])
    corpus_ids = deep_seq_map(corpus_ids, lambda xs: len(xs), keys=['question', 'support'], fun_name='lengths', expand=True)
    return corpus_ids, vocab, target_vocab, candidate_vocab


def main():
    # this is where the list of all models lives, add those if they work
    reader_models = {
        'bicond_singlesupport_reader': ReaderModel.conditional_reader_model,
        'boe': ReaderModel.boe_reader_model,
        #'log_linear': ReaderModel.create_log_linear_reader,
        #'model_f': ReaderModel.create_model_f_reader,
        #'boe': ReaderModel.create_bag_of_embeddings_reader
    }

    support_alts = {'none', 'single', 'multiple'}
    question_alts = answer_alts = {'single', 'multiple'}
    candidate_alts = {'open', 'per-instance', 'fixed'}

    #todo clean up
    #common default input files - for rapid testing
    """
    train_default = 'data/SQuAD/snippet_quebapformat.json'
    dev_default = 'data/sentihood/single_quebap.json'
    test_default = 'data/sentihood/single_quebap.json'
    """
    train_default = "./quebap/data/SNLI/snli_1.0/snli_1.0_train_quebap_v1.json"
    dev_default = "./quebap/data/SNLI/snli_1.0/snli_1.0_dev_quebap_v1.json"
    test_default = "./quebap/data/SNLI/snli_1.0/snli_1.0_test_quebap_v1.json"


    #args
    parser = argparse.ArgumentParser(description='Train and Evaluate a machine reader')
    parser.add_argument('--debug', default=True, type=bool, help="Run in debug mode, in which case the training file is also used for testing")
    parser.add_argument('--debug_examples', default=2000, type=int, help="If in debug mode, how many examples should be used")
    parser.add_argument('--train', default=train_default, type=argparse.FileType('r'), help="Quebap training file")
    parser.add_argument('--dev', default=dev_default, type=argparse.FileType('r'), help="Quebap dev file")
    parser.add_argument('--test', default=test_default, type=argparse.FileType('r'), help="Quebap test file")
    parser.add_argument('--supports', default='single', choices=sorted(support_alts), help="None, single or multiple supporting statements per instance")
    parser.add_argument('--questions', default='single', choices=sorted(question_alts), help="None, single or multiple questions per instance")
    parser.add_argument('--candidates', default='fixed', choices=sorted(candidate_alts), help="Open, per-instance or fixed candidates")
    parser.add_argument('--answers', default='single', choices=sorted(answer_alts), help="Open, per-instance or fixed candidates")
    parser.add_argument('--batch_size', default=5, type=int, help="Batch size for training data")
    parser.add_argument('--dev_batch_size', default=5, type=int, help="Batch size for development data")
    parser.add_argument('--repr_dim_input', default=5, type=int, help="Size of the input representation (embeddings)")
    parser.add_argument('--repr_dim_output', default=5, type=int, help="Size of the output representation")
    parser.add_argument('--pretrain', default=False, type=bool, help="Use pretrained embeddings, by default the initialisation is random")
    parser.add_argument('--train_pretrain', default=False, type=bool,
                        help="Continue training pretrained embeddings together with model parameters")
    parser.add_argument('--normalize_pretrain', default=True, type=bool,
                        help="Normalize pretrained embeddings")
    parser.add_argument('--model', default='bicond_singlesupport_reader', choices=sorted(reader_models.keys()), help="Reading model to use")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--epochs', default=3, type=int, help="Number of epochs to train for")
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


    bucket_order = ('question','support') #composite buckets; first over question, then over support
    bucket_structure = (4,4) #will result in 16 composite buckets, evenly spaced over questions and supports

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
        train_data, dev_data, test_data = [quebap_load(name) for name in [args["train"], args["dev"], args["test"]]]
        print('loaded train/dev/test data')
        if args.pretrain:
            # emb_file = 'GoogleNews-vectors-negative300.bin.gz'
            # embeddings = load_embeddings(path.join('quebap', 'data', 'word2vec', emb_file),'word2vec')
            emb_file = 'glove.840B.300d.zip'
            embeddings = load_embeddings(path.join('quebap', 'data', 'GloVe', emb_file), 'glove')
            print('loaded pre-trained embeddings (%s)'%emb_file)

    emb = embeddings.get if args.pretrain else None

    checkpoint()
    print('encode train data')
    train_data, train_vocab, train_answer_vocab, train_candidate_vocab = pipeline(train_data, emb=emb)
    N_oov = train_vocab.count_oov()
    N_pre = train_vocab.count_pretrained()
    print('In Training data vocabulary: %d pre-trained, %d out-of-vocab.' % (N_pre, N_oov))

    vocab_size = len(train_vocab)
    answer_size = len(train_answer_vocab)
    candidate_size = len(train_candidate_vocab)

    # @todo: we should allow to set vocab_size for smaller vocab

    # this is a bit of a hack since args are supposed to be user-defined, but it's cleaner that way with passing on args to reader models
    parser.add_argument('--vocab_size', default=vocab_size, type=int)
    parser.add_argument('--answer_size', default=answer_size, type=int)
    args = parser.parse_args()

    print("\tvocab size:  %d" % vocab_size)
    print("\tanswer size: %d" % answer_size)
    print("\tcandidate size: %d" % candidate_size)

    checkpoint()
    print('encode dev data')
    dev_data, _, _, _ = pipeline(dev_data, train_vocab, train_answer_vocab, train_candidate_vocab, freeze=True)
    checkpoint()
    print('encode test data')
    test_data, _, _, _ = pipeline(test_data, train_vocab, train_answer_vocab, train_candidate_vocab, freeze=True)
    checkpoint()

    print('build NeuralVocab')
    #todo: foresee input args to set these parameters
    nvocab = NeuralVocab(train_vocab, input_size=args.repr_dim_input, use_pretrained=args.pretrain,
                         train_pretrained=args.train_pretrain, unit_normalize=args.normalize_pretrain)

    checkpoint()
    print('build model %s'%args.model)
    reader = ReaderModel()
    (logits, loss, predict), placeholders = reader_models[args.model](reader, nvocab, **vars(args))

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
        AccuracyHook(dev_feed_dicts, predict, placeholders['answers'], 2)
    ]

    train(loss, optim, train_feed_dicts, max_epochs=args.epochs, hooks=hooks)



if __name__ == "__main__":
    main()
