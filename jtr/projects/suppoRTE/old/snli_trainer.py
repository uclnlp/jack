"""
adapted from Dirk Weissenborn's train_snli
"""

import numpy as np
import random
import sys
import tensorflow as tf
from time import time
import math
import json
import functools
from sklearn.utils import shuffle

from jtr.projects.suppoRTE.prepare_data import load_data
from jtr.projects.suppoRTE.rte_models import rte_model


def encode_labels(labels):
    """
    convert list of string labels ('neutral','entailment', or 'contradiction') to numerical vector
    """
    Y = np.zeros([len(labels)]).astype('int64')
    for j, y in enumerate(labels):
        if y == 'neutral':
            Y[j] = 0
        elif y == 'entailment':
            Y[j] = 1
        else:
            Y[j] = 2
    return Y


#create batch given example sentences
def batchify(batchA, batchB, idsA, idsB, lengthsA, lengthsB, max_length=None, max_batch_size=None):
    #TODO: why pass these matrices each time? 
    #TODO: rewrite more efficiently
    """
    :param batchA: list of lists with indices, for instances (premise sentences) in the selected batch
    :param batchB: list of lists with indices, for instances (hypothesis sentences) in the selected batch
    :param idsA, idsB: ndarrays with shape (max_length x max_batch_size)  
    :param lenghtsA, lengthsB: numpy with lengths of instances in batchA, batchB 
    :param max_length: max sequence length
    :param max_batch_size
    :returns idsA (ndarray with shape max_length, max_batch_size) with word id's. Beyond the actual sequence length: meaningless numbers (at first 1's, later on left-overs from earlier batches)
    """
    idsA = np.ones([max_length, max_batch_size]) if idsA is None else idsA
    idsB = np.ones([max_length, max_batch_size]) if idsB is None else idsB

    lengthsA = np.zeros([max_batch_size], np.int32) if lengthsA is None else lengthsA
    lengthsB = np.zeros([max_batch_size], np.int32) if lengthsB is None else lengthsB

    for i in range(len(batchA)):
        lengthsA[i] = len(batchA[i])
        for j in range(len(batchA[i])):
            idsA[j][i] = batchA[i][j] #first dimension: sequence, 2nd dim: instance

    for i in range(len(batchB)):
        lengthsB[i] = len(batchB[i])
        for j in range(len(batchB[i])):
            idsB[j][i] = batchB[i][j]

    return idsA, idsB, lengthsA, lengthsB




def evaluate(model, sess, dsA, dsB, labels, batch_size, max_length):
    idsA, idsB, lengthsA, lengthsB = None, None, None, None
    e_off = 0
    num_correct = 0.0
    y = encode_labels(labels)
    #op_weights_monitor = {ops[int(w.name[-3:-2])]:[] for w in op_weights}

    while e_off < len(dsA):
        idsA, idsB, lengthsA, lengthsB = batchify(dsA[e_off:e_off + batch_size],
                                                  dsB[e_off:e_off + batch_size],
                                                  idsA, idsB, lengthsA, lengthsB,
                                                  max_length=max_length,
                                                  max_batch_size=batch_size)
        size = min(len(dsA) - e_off, batch_size)#actual batch_size used (in case at end of epoch)
        #allowed_conds = ["/cond_%d/" % (2*i) for i in range(min(np.min(lengthsA), np.min(lengthsB)))]
        #current_weights = [w for w in op_weights if any(c in w.name for c in allowed_conds)]
        result = sess.run([model["probs"]] , feed_dict={model["idsA"]: idsA[:,:size],
                                                        model["idsB"]: idsB[:,:size],
                                                        model["lengthsA"]: lengthsA[:size],
                                                        model["lengthsB"]: lengthsB[:size]})
        #result: tf.nn.softmax(scores):  
        num_correct += np.sum(np.equal(np.argmax(result[0], axis=1), y[e_off:e_off+size]))# np.argmax(result[0], axis=1): results in batch_size=length vector with predicted label indices  
        
        e_off += size

        #for probs, w in zip(result[1:], current_weights):
        #    op_weights_monitor[ops[int(w.name[-3:-2])]].extend(probs.tolist())

    #for k,v in op_weights_monitor.items():
    #    hist, _ = np.histogram(np.array(v), bins=5,range=(0.0,1.0))
    #    hist = (hist * 1000) / np.sum(hist)
    #    print(k, hist.tolist())

    accuracy = num_correct/e_off
    return accuracy




def train(embeddings, FLAGS):
    # Load data
    print("Preparing data...")
    t0 = time()
    embedding_size = embeddings.vectors.shape[1]
    trainA, trainB, devA, devB, testA, testB, labels, vocab, oo_vocab = load_data(FLAGS.data, embeddings)
    #trainA, trainB, devA, devB, testA, testB, [trainS, devS, testS], vocab, oo_vocab
    #trainA : list (dataset) of lists (premise sentences) with symbol ID's (indices in embedding tensor)


    # embeddings
    task_embeddings = np.random.normal(size=[len(vocab)+len(oo_vocab), embedding_size]).astype("float32")
    for w, i in vocab.items():
        #within vocab terms: initialize with correct embedding
        task_embeddings[len(oo_vocab) + i] = embeddings[w]
        #indices within task_embeddings: first oo_vocab, then vocab 

    # accumulate counts for buckets
    def max_length(sentences, max_l=0):
        for s in sentences:
            l = len(s)
            max_l = max(l, max_l)
        return max_l

    max_l = max_length(trainA)
    max_l = max_length(trainB, max_l)
    max_l = max_length(devA, max_l)
    max_l = max_length(devB, max_l)
    max_l = max_length(testA, max_l)
    max_l = max_length(testB, max_l)

    print("Done (%ds)."%(time()-t0))

    l2_lambda = FLAGS.l2_lambda
    learning_rate = FLAGS.learning_rate
    h_size = FLAGS.h_size
    mem_size = FLAGS.mem_size
    rng = random.Random(FLAGS.seed)
    batch_size = FLAGS.batch_size

    accuracies = []

    idsA, idsB, lengthsA, lengthsB = None, None, None, None

    for run_id in range(FLAGS.runs):
        tf.reset_default_graph()#Clears the default graph stack and resets the global default graph. The default graph is a property of the current thread. This function applies only to the current thread. 
        config = tf.ConfigProto(allow_soft_placement=True)#let tf automatically select device if specified device doesn't exist 
        config.gpu_options.allow_growth = True# do not assign all gpu memory from the start; increase as needed during run
        with tf.Session(config=config) as sess:
            tf.set_random_seed(rng.randint(0, 10000))#as such: depends on rng seed set at the start, but different for each run_id 
            rng2 = random.Random(rng.randint(0, 10000))

            cellA = cellB = None
            if FLAGS.cell == 'LSTM':
                cellA = cellB = tf.nn.rnn_cell.BasicLSTMCell(mem_size,state_is_tuple=True)
            elif FLAGS.cell == 'GRU':
                cellA = cellB = tf.nn.rnn_cell.GRUCell(mem_size)

            #all tunable:
            tunable_embeddings, fixed_embeddings = task_embeddings, None
            #only out-of-vocabulary are tunable
            if FLAGS.embedding_mode == "fixed":
                tunable_embeddings, fixed_embeddings = task_embeddings[:len(oo_vocab)], task_embeddings[len(oo_vocab):]
            
            with tf.device(FLAGS.device):
                model = rte_model(max_l, l2_lambda, learning_rate, h_size, cellA, cellB, tunable_embeddings, fixed_embeddings, FLAGS.keep_prob)

            tf.get_variable_scope().reuse_variables()#keep Variables 

            #op_weights = [w.outputs[0] for w in tf.get_default_graph().get_operations()
            #              if not "grad" in w.name and w.name[:-2].endswith("op_weight")]
            #get_operations(): gets operations defined on the graph; each operation represents a graph node that performs computation on tensors.
            #tf.Operation.outputs: list of tensor objects representing the outputs of the considered op
            #tf.Operation.name: full name of the operation

            saver = tf.train.Saver(tf.trainable_variables())
            sess.run(tf.initialize_all_variables())
            num_params = functools.reduce(lambda acc, x: acc + x.size, sess.run(tf.trainable_variables()), 0)
            #reduce(function, iterable, initializer): apply function cumulatively to the items of iterable, to reduce it to a single value; start value at initializer 
            print("Num params: %d" % num_params)
            print("Num params (without embeddings): %d" % (num_params - (len(oo_vocab) + len(vocab)) * embedding_size))

            shuffledA, shuffledB, y = \
                shuffle(list(trainA), list(trainB), list(labels[0]), random_state=rng2.randint(0, 1000))

            offset = 0
            loss = 0.0
            epochs = 0
            i = 0
            accuracy = float("-inf")
            step_time = 0.0
            epoch_acc = 0.0
            while not FLAGS.eval:# do until break, unless FLAGS.eval is True 
                start_time = time()
                idsA, idsB, lengthsA, lengthsB = batchify(shuffledA[offset:offset+batch_size],
                                                          shuffledB[offset:offset+batch_size],
                                                          idsA, idsB, lengthsA, lengthsB,
                                                          max_length=max_l,
                                                          max_batch_size=batch_size)
                train_labels = encode_labels(y[offset:offset+batch_size])
                # update initialized embeddings only after first epoch
                #TODO: why not later / from the start?
                update = model["update"] if epochs>= 1 else model["update_ex"] #update op
                l, _ = sess.run([model["loss"], update],
                                feed_dict={model["idsA"]:idsA,
                                           model["idsB"]:idsB,
                                           model["lengthsA"]: lengthsA,
                                           model["lengthsB"]: lengthsB,
                                           model["y"]:train_labels})

                offset += batch_size
                loss += l
                i += 1
                step_time += (time() - start_time)

                sys.stdout.write("\r%.1f%% Loss: %.3f" % ((i*100.0) / FLAGS.checkpoint, loss / i))
                sys.stdout.flush()

                #if end of current epoch: reset batches, shuffle, and evaluate on dev-set
                if offset + batch_size > len(shuffledA):
                    epochs += 1
                    shuffledA, shuffledB, y = shuffle(shuffledA, shuffledB, y, random_state=rng2.randint(0, 1000))
                    offset = 0
                    sess.run(model["keep_prob"].assign(1.0))
#                    def evaluate(model, sess, dsA, dsB, labels, batch_size, max_length):
                    acc = evaluate(model, sess, devA, devB, labels[1], batch_size, max_l)
                    sess.run(model["keep_prob"].initializer) #set back to original keep_prob
                    print("\n%d epochs done! Accuracy on Dev: %.3f" % (epochs, acc))
                    if acc < epoch_acc + 1e-3: #if dev-set accuracy hasn't increased enough since previous epoch
                        print("Decaying learning-rate!")
                        lr = tf.get_variable("model/lr")
                        sess.run(lr.assign(lr * FLAGS.learning_rate_decay))
                    epoch_acc = acc

                if i == FLAGS.checkpoint: #e.g., every 1000 batches (over epochs): measure loss, evaluate on dev-set
                    loss /= i
                    sess.run(model["keep_prob"].assign(1.0))
                    acc = evaluate(model, sess, devA, devB, labels[1], batch_size, max_l)
                    sess.run(model["keep_prob"].initializer)

                    print("\nTrain loss: %.3f, Accuracy on Dev: %.3f, Step Time: %.3f, Total Time: %.2fh" % (loss, acc, step_time/i, (time()-t0)/3600.))
                    i = 0
                    step_time = 0.0
                    loss = 0.0
                    if acc > accuracy + 1e-5:
                        accuracy = acc
                        saver.save(sess, FLAGS.model_path)
                    else: #accuracy on dev-set no longer increases 
                        if epochs >= FLAGS.min_epochs:
                            break

            #final result of this run:
            saver.restore(sess, FLAGS.model_path)
            sess.run(model["keep_prob"].assign(1.0))
            acc = evaluate(testA, testB, labels[2])
            accuracies.append(acc)
            print('######## Run %d #########' % run_id)
            print('Test Accuracy: %.4f' % acc)
            print('########################')

    #average over all FLAGS.runs:
    mean_accuracy = sum(accuracies) / len(accuracies)

    def s_dev(mean, pop):
        d = 0.0
        for el in pop:
            d += (mean-el) * (mean-el)
        return math.sqrt(d/len(pop))

    print('######## Overall #########')
    print('Test Accuracy: %.4f (+-%.4f)' % (mean_accuracy,  s_dev(mean_accuracy, accuracies)))
    print('########################')

    if FLAGS.result_file:
        with open(FLAGS.result_file, 'w') as f:
            f.write('Accuracy: %.4f (%.4f)\n' % (mean_accuracy,  s_dev(mean_accuracy, accuracies)))
            f.write("Configuration: \n")
            f.write(json.dumps(FLAGS.__flags, sort_keys=True, indent=2, separators=(',', ': ')))

    return mean_accuracy





