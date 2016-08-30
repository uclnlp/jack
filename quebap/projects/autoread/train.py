import os
import random
import time
from quebap.projects.autoread import *
import tensorflow as tf
import sys
import functools
import web.embeddings as embeddings
import numpy as np

# data loading specifics
from quebap.projects.autoread.autoreader import AutoReader
from quebap.projects.autoread.wikireading import load_vocab

tf.app.flags.DEFINE_string('data', None, 'Path to extracted TFRecord data. Assuming contains document.vocab')
tf.app.flags.DEFINE_string("trainset_prefix", "test-00000-of-00015", "Comma separated datasets to train on.")
tf.app.flags.DEFINE_string("devset_prefix", "test-00001-of-00015", "Development set")
#tf.app.flags.DEFINE_string("testset_prefix", "test", "Test set.")

# model
tf.app.flags.DEFINE_integer("size", 256, "hidden size of model")
tf.app.flags.DEFINE_integer("max_vocab", -1, "maximum vocabulary size")
tf.app.flags.DEFINE_string("composition", 'GRU', "'LSTM', 'GRU'")

#training
tf.app.flags.DEFINE_float("dropout", 0.2, "Dropout.")
tf.app.flags.DEFINE_float("cloze_dropout", 0.8, "Dropout for token to predict.")
tf.app.flags.DEFINE_float("learning_rate", 1e-2, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay when loss on validation set does not improve.")
tf.app.flags.DEFINE_integer("batch_size", 25, "Number of examples in each batch for training.")
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")
tf.app.flags.DEFINE_integer("max_iterations", -1, "Maximum number of batches during training. -1 means until convergence")
tf.app.flags.DEFINE_integer("ckpt_its", 1000, "Number of iterations until running checkpoint. Negative means after every epoch.")
tf.app.flags.DEFINE_integer("random_seed", 1234, "Seed for rng.")
tf.app.flags.DEFINE_integer("min_epochs", 1, "Minimum number of epochs.")
tf.app.flags.DEFINE_string("save_dir", "save/" + time.strftime("%d%m%Y_%H%M%S", time.localtime()),
                           "Where to save model and its configuration, always last will be kept.")
tf.app.flags.DEFINE_string("init_model_path", None, "Path to model to initialize from.")
tf.app.flags.DEFINE_string("embeddings", None, "Init with word embeddings from given path in w2v binary format.")
tf.app.flags.DEFINE_string("max_context_length", 300, "Maximum length of context.")
tf.app.flags.DEFINE_string("dataset", "wikireading", "dataset on which we want to train")


FLAGS = tf.app.flags.FLAGS

random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

word_ids, vocab = load_vocab(os.path.join(FLAGS.data, "document.vocab"))
if FLAGS.max_vocab < 0:
    FLAGS.max_vocab = len(vocab)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    print("Preparing Samplers ...")
    train_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.trainset_prefix)]
    random.shuffle(train_fns)
    print("Training sets: ", train_fns)
    sampler = sampler_for(FLAGS.dataset)(sess, FLAGS.data, train_fns, FLAGS.batch_size, max_vocab=FLAGS.max_vocab,
                           max_answer_vocab=FLAGS.max_vocab,
                           max_length=FLAGS.max_context_length, vocab=word_ids)

    train_dir = os.path.join(FLAGS.save_dir)
    dev_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.devset_prefix)]
    print("Valid sets: ", dev_fns)
    valid_sampler = sampler_for(FLAGS.dataset)(sess, FLAGS.data, dev_fns, FLAGS.batch_size, max_vocab=FLAGS.max_vocab,
                                 max_answer_vocab=FLAGS.max_vocab,
                                 max_length=FLAGS.max_context_length, vocab=word_ids)
    #test_fns = [fn for fn in os.listdir(FLAGS.data) if fn.startswith(FLAGS.testset_prefix)]
    #print("Test sets: ", test_fns)
    #test_sampler = BatchSampler(sess, FLAGS.data, test_fns, FLAGS.batch_size, max_vocab=FLAGS.max_vocab,
    #                            max_answer_vocab=FLAGS.max_vocab,
    #                            max_length=FLAGS.max_context_length, vocab=word_ids)

    devices = FLAGS.devices.split(",")
    m = AutoReader(FLAGS.size, FLAGS.max_vocab, FLAGS.max_context_length,
                   learning_rate=FLAGS.learning_rate, devices=devices,
                   keep_prob=1.0-FLAGS.dropout, cloze_keep_prob=1.0 - FLAGS.cloze_dropout, composition=FLAGS.composition)

    print("Created model!")

    best_path = []
    checkpoint_path = os.path.join(train_dir, "model.ckpt")

    previous_loss = list()
    epoch = 0

    if FLAGS.init_model_path:
        print("Loading from path " + FLAGS.init_model_path)
        m.model_saver.restore(sess, FLAGS.init_model_path)
    elif os.path.exists(train_dir) and any("ckpt" in x for x in os.listdir(train_dir)):
        newest = max(map(lambda x: os.path.join(train_dir, x),
                         filter(lambda x: not x.endswith(".meta") and "ckpt" in x, os.listdir(train_dir))),
                     key=os.path.getctime)
        print("Loading from checkpoint " + newest)
        m.all_saver.restore(sess, newest)
    else:
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        print("Initializing variables ...")
        sess.run(tf.initialize_all_variables())
        if FLAGS.embeddings is not None:
            print("Init embeddings with %s..." % FLAGS.embeddings)
            e = embeddings.load_embedding(FLAGS.embeddings)
            em = sess.run(m.input_embeddings)
            for j in range(FLAGS.max_vocab):
                w = vocab[j]
                v = e.get(w)
                if v is not None:
                    em[j, :v.shape[0]] = v
            sess.run(m.input_embeddings.assign(em))

    num_params = functools.reduce(lambda acc, x: acc + x.size, sess.run(tf.trainable_variables()), 0)
    print("Num params: %d" % num_params)

    print("Initialized model.")

    def validate():
        # Run evals on development set and print(their perplexity.)
        print("########## Validation ##############")
        e = valid_sampler.epoch
        l = 0.0
        ctr = 0
        sess.run(m.cloze_keep_prob.set(0.0))
        while valid_sampler.epoch == e:
            l += m.run(sess, m.loss, valid_sampler.get_batch())
            ctr += 1
            sys.stdout.write("\r%d - %.3f" % (ctr, l /ctr))
            sys.stdout.flush()
        sess.run(m.cloze_keep_prob.initializer())
        l /= ctr
        print("loss: %.3f" % l)
        print("####################################")

        if not best_path or l < min(previous_loss):
            if best_path:
                best_path[0] = m.all_saver.save(sess, checkpoint_path, global_step=m.global_step, write_meta_graph=False)
            else:
                best_path.append(m.all_saver.save(sess, checkpoint_path, global_step=m.global_step, write_meta_graph=False))

        if previous_loss and l > previous_loss[-1]:
            # if no significant improvement decay learningrate
            print("Decaying learningrate.")
            sess.run(m.learning_rate.assign(m.learning_rate * FLAGS.learning_rate_decay))

        previous_loss.append(l)
        return l

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)

    loss = 0.0
    step_time = 0.0
    ckpt_result = float("inf")
    i = 0
    while True:
        i += 1
        start_time = time.time()
        batch = sampler.get_batch()
        # already fetch next batch parallel to running model
        loss += m.run(sess, [m.update, m.loss], batch)[1]
        step_time += (time.time() - start_time)

        sys.stdout.write("\r%.1f%% Loss: %.3f" % (i*100.0 / FLAGS.ckpt_its, loss / i))
        sys.stdout.flush()

        if i % FLAGS.ckpt_its == 0:
            i = 0
            loss /= FLAGS.ckpt_its
            print("")
            step_time /= FLAGS.ckpt_its
            print("global step %d learning rate %.5f, step-time %.3f, loss %.4f" % (m.global_step.eval(),
                                                                                    m.learning_rate.eval(),
                                                                                    step_time, loss))
            step_time, loss = 0.0, 0.0
            result = validate()
            if result > ckpt_result + 1e-4:
                print("Stop learning!")
                break
            else:
                ckpt_result = result

    best_valid_loss = max(previous_loss) if previous_loss else 0.0
   # print("Restore model to best on validation, with Accuracy: %.3f" % best_valid_acc)
    m.all_saver.restore(sess, best_path[0])
    model_name = best_path[0].split("/")[-1]
    m.model_saver.save(sess, os.path.join(train_dir, "final_model.tf"), write_meta_graph=False)
   # print("########## Test ##############")
   # MAP = eval.eval_dataset(sess, m, test_sampler, True)
   # print("MAP: %.3f" % MAP)
   # print("##############################")
