'''Joint training all/almost all tasks'''
from __future__ import absolute_import
from __future__ import print_function

from data_utils_jtr import load_task, vectorize_data
from sklearn import cross_validation, metrics
from memn2n.memn2n import MemN2N
from itertools import chain
import itertools

import tensorflow as tf
import numpy as np
from functools import reduce
import os

tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")  #try something between 5 and 40
tf.flags.DEFINE_integer("evaluation_interval", 2, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 50, "Batch size for training.")
tf.flags.DEFINE_integer("num_hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 200, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("vocab_size", 100000, "Vocabulary size")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("train_file", "../../data/SQuAD/snippet_jtrformat.json", "Training file containing QA dataset. Required annotation: questions with answer spans and supporting text.")
tf.flags.DEFINE_string("test_file", "../../data/SQuAD/snippet_jtrformat.json", "Test file containing QA dataset. Required annotation: questions with answer spans and supporting text.")
tf.flags.DEFINE_string("dev_file", "../../data/SQuAD/snippet_jtrformat.json", "Dev file containing QA dataset. Required annotation: questions with answer spans and supporting text.")
FLAGS = tf.flags.FLAGS
savepath = "./out/"

# task data
train, dev, test = load_task(FLAGS.train_file, FLAGS.test_file, FLAGS.dev_file)

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in train)))


word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
#print(word_idx)

# overwrite some the default values in case vocab is smaller, story size and memory size is larger etc
vocab_size = len(word_idx) + 1 # +1 for nil word
max_story_size = max(map(len, (s for s, _, _ in train + test)))
mean_story_size = int(np.mean([len(s) for s, _, _ in train + test])) #mean_story_size = int(np.mean(map(len, (s for s, _, _ in train + test))))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in train + test)))
query_size = max(map(len, (q for _, q, _ in train + test)))
sentence_size = max(query_size, sentence_size)
memory_size = min(100, max_story_size)

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
trainS, trainQ, trainA = vectorize_data(train, word_idx, sentence_size, memory_size)
valS, valQ, valA = vectorize_data(dev, word_idx, sentence_size, memory_size)
testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)



tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size
optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
with tf.Session() as sess:
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, session=sess,   # max_gradient_norm=FLAGS.max_gradient_norm,
                   hops=FLAGS.num_hops, optimizer=optimizer)

    saver = tf.train.Saver(tf.trainable_variables())

    valid_cost_prev = float('inf')
    for t in range(1, FLAGS.num_epochs+1):
        total_cost = 0.0
        valid_cost = 0.0
        for start in range(0, n_train, batch_size):
            end = start + batch_size
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            cost_t = model.batch_fit(s, q, a)
            total_cost += cost_t

        for start in range(0, n_val, batch_size):
            end_v = start + batch_size
            s_v = valS[start:end]
            q_v = valQ[start:end]
            a_v = valA[start:end]
            cost_v = model.batch_fit(s_v, q_v, a_v)
            valid_cost += cost_v

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                pred = model.predict(s, q)
                train_preds += list(pred)

            val_preds = model.predict(valS, valQ)
            train_acc = metrics.accuracy_score(train_labels, np.array(train_preds))
            val_acc = metrics.accuracy_score(val_labels, np.array(val_preds))

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Validation Cost:', valid_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)

            if valid_cost > valid_cost_prev:
                print("Dev loss bigger than last epoch! Stopping training early.")
                break
            valid_cost_prev = valid_cost

            print("Saving model...")
            path = "save/test_ep" + str(t)
            if not os.path.exists(path):
                os.makedirs(path)
            saver.save(sess, os.path.join(path, "model.tf"))

            print('-----------------------')

    print('-----------------------')
    print("Testing instances:")
    print('-----------------------')
    i = 0
    for _, query, answer in test:
        print(test_labels[i], answer, query)
        i += 1

    print('-----------------------')

    test_preds = model.predict(testS, testQ)
    test_acc = metrics.accuracy_score(test_labels, test_preds)
    #p_r_f = metrics.classification_report(test_labels, test_preds)
    print("Testing labels", test_labels)
    print("Testing predictions", test_preds)
    print("Testing Accuracy:", test_acc)
    #print("Testing P/R/F1:", p_r_f)

