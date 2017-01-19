# -*- coding: utf-8 -*-

import argparse
from .models import *

from time import time


class Duration(object):
    def __init__(self):
        self.t0 = time()
        self.t = time()
    def __call__(self):
        print('Time since last checkpoint : %.2fmin'%((time()-self.t)/60.))
        self.t = time()

checkpoint = Duration()


def train_reader(reader: MultipleChoiceReader, train_data, test_data, num_epochs, batch_size,
                 optimiserType="GradientDescent", gradDebug=False, use_train_generator_for_test=False):
    """
    Train a reader, and test on test set.
    :param reader: The reader to train
    :param train_data: the jtr training file
    :param test_data: the jtr test file
    :param num_epochs: number of epochs to train
    :param batch_size: size of each batch
    :param optimiser: the optimiser to use
    :return: Nothing
    """

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, 0.96, staircase=True)

    if optimiserType == "GradientDescent":
        optimiser = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        optimiser = tf.train.RMSPropOptimizer(learning_rate)

    grads = optimiser.compute_gradients(reader.loss)

    capped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
    opt_op = optimiser.apply_gradients(capped_grads, global_step=global_step)


    #opt_op = optimiser.minimize(reader.loss)

    sess = tf.Session()
    #writer = tf.train.SummaryWriter("log", sess.graph)
    sess.run(tf.initialize_all_variables())

    step = 0

    for epoch in range(0, num_epochs):
        checkpoint()
        print("Epoch:", epoch)
        avg_loss = 0
        count = 0
        #predictions_tr = []
        i = 0
        for batch in reader.tensorizer.create_batches(train_data, batch_size=batch_size):
            #print(batch)

            # compute gradients
            """if gradDebug == True and i == 0:
                grad_vals = sess.run((grads), feed_dict=batch)
                print('some grad_vals: ', grad_vals[0])"""

            #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #run_metadata = tf.RunMetadata()

            # applies the gradients
            #_, loss = sess.run((opt_op, reader.loss), feed_dict=batch, options=run_options, run_metadata=run_metadata)
            _, loss = sess.run((opt_op, reader.loss), feed_dict=batch)

            #writer.add_run_metadata(run_metadata, 'step%d' % step)
            #writer.add_summary(summary, step)
            avg_loss += loss
            count += 1
            #if count % 2 == 0:
            #    print("Avg Loss: {}".format(avg_loss / count))

            #scores = sess.run(reader.scores, feed_dict=batch)
            #candidates_ids = batch[reader.tensorizer.candidates]
            #predictions_tr += reader.tensorizer.convert_to_predictions(batch, scores)

            i += 1
            step += 1



        print("Global step: ", global_step.eval(session=sess))
        print("Learning rate: ", learning_rate.eval(session=sess))

        print("Train Loss: ", np.sum(avg_loss) / count)
        #print("Acc: ", accuracy(train_data, {'instances': predictions_tr}))
        #print("Train MRR@5: ", mrr_at_k(train_data, {'instances': predictions_tr}, 5))

        #predictions_test = []
        #i = 0
        #for batch_test in reader.tensorizer.create_batches(test_data, test=not use_train_generator_for_test, batch_size=1):
            #print(i)
            #print(batch_test)
        #    try:
        #        scores = sess.run(reader.scores, feed_dict=batch_test)
                #candidates_ids = batch_test[reader.tensorizer.candidates]
        #        predictions_test += reader.tensorizer.convert_to_predictions(batch_test, scores)
        #    except ValueError:
                #print("ValueError for", batch_test[reader.tensorizer.questions])
        #        continue
            #i += 1

        #print("Test MRR@5: ",
        #      mrr_at_k(test_data, {'instances': predictions_test}, 5, print_details=False))  # was: accuracy, accuracy_multi
        #print("Test Acc: ", accuracy(test_data, {'instances': predictions_test}))

    print("Finished training, predictions on test:")
    predictions_test = []
    for batch_test in reader.tensorizer.create_batches(test_data, test=not use_train_generator_for_test, batch_size=1):
        try:
            scores = sess.run(reader.scores, feed_dict=batch_test)
            #candidates_ids = batch[reader.tensorizer.candidates]
            #print(candidates_ids)
            predictions_test += reader.tensorizer.convert_to_predictions(batch_test, scores)
        except ValueError:
            print("ValueError for", batch_test[reader.tensorizer.questions])
            continue

    print("Test MRR@5: ", mrr_at_k(test_data, {'instances': predictions_test}, 5, print_details=True))  # was: accuracy, accuracy_multi


def main():
    readers = {
        'se': create_sequence_embeddings_reader,
        'bowv': create_bowv_embeddings_reader,  # this is with mm between que and mean averaged sup
        'bowv_concat': create_bowv_concat_embeddings_reader, # this is with concat between que and mean averaged sup
        'bowv_nosupport': create_bowv_nosupport_embeddings_reader
    }

    parser = argparse.ArgumentParser(description='Train and Evaluate a machine reader')
    #parser.add_argument('--trainKBP', default='../../data/scienceQA/scienceQA_kbp_all.json', type=argparse.FileType('r'), help="jtr training file")
    parser.add_argument('--trainCloze', default='../../data/scienceQA/scienceQA_cloze_with_support_2016-11-03.json',#scienceQA_cloze_withcont_2016-10-25_small.json',#scienceQA_cloze_with_support_2016-11-03.json',#scienceQA_cloze_withcont_2016-10-9.json',
                    type=argparse.FileType('r'), help="jtr training file")
    parser.add_argument('--testSetup', default='clozeOnly', help="clozeOnly, kbpOnly, kbpForTest, clozeForTest, both")
    #parser.add_argument('--test', default='../../data/scienceQA/scienceQA_kbp_all_nosupport.json', type=argparse.FileType('r'), help="jtr test file")
    parser.add_argument('--batch_size', default=20, type=int, metavar="B", help="Batch size (suggestion)")
    parser.add_argument('--repr_dim', default=50, type=int, help="Size of the hidden representation")
    parser.add_argument('--support_dim', default=50, type=int, help="Size of the hidden representation for support")
    parser.add_argument('--model', default='bowv_concat', choices=sorted(readers.keys()), help="Reading model to use")
    parser.add_argument('--epochs', default=20, type=int, help="Number of epochs to train for")


    args = parser.parse_args()

    checkpoint()

    #reading_dataset = shorten_reading_dataset(json.load(args.train), args.train_begin, args.train_end)

    if "both" in args.testSetup or "kbp" in args.testSetup:
        reading_dataset = json.load(args.trainKBP)
        print("Reading kbp file done!")
        print("Number reading instances:", len(reading_dataset['instances']))

        for ii, inst in enumerate(reading_dataset['instances']):
            print(inst['questions'][0]['question'], inst['questions'][0]['answers'], len(inst['support']))

        # reading_dataset = shorten_candidate_list(reading_dataset)

    if "both" in args.testSetup or "cloze" in args.testSetup:
        reading_dataset_cloze = json.load(args.trainCloze)
        print("Reading cloze file done!")
        print("Number reading instances:", len(reading_dataset_cloze['instances']))

        #for ii, inst in enumerate(reading_dataset_cloze['instances']):
        #    print(inst['questions'][0]['question'], inst['questions'][0]['answers'], len(inst['support']))

        # reading_dataset = shorten_candidate_list(reading_dataset)

    if args.testSetup == "both":
        training_dataset = shorten_reading_dataset(reading_dataset, 101, len(reading_dataset['instances']) - 1)
        testing_dataset = shorten_reading_dataset(reading_dataset, 0, 100)

        result = copy.copy(training_dataset)
        result['instances'] = training_dataset['instances'] + reading_dataset_cloze['instances']
        training_dataset = result

    elif args.testSetup == "clozeOnly":
        #training_dataset = shorten_reading_dataset(reading_dataset_cloze, 100001, len(reading_dataset_cloze['instances']) - 1)
        #testing_dataset = shorten_reading_dataset(reading_dataset_cloze, 0, 100000)
        #testing_dataset = shorten_reading_dataset(reading_dataset_cloze, 362001, len(reading_dataset_cloze['instances']) - 1)
        #training_dataset = shorten_reading_dataset(reading_dataset_cloze, 0, 2000)
        testing_dataset = shorten_reading_dataset(reading_dataset_cloze, 1001, 2000)
        training_dataset = shorten_reading_dataset(reading_dataset_cloze, 0, 1000)
        #training_dataset = reading_dataset_cloze
        #testing_dataset = reading_dataset_cloze


    #training_dataset = shorten_reading_dataset(reading_dataset, 0, 12)
    #testing_dataset = shorten_reading_dataset(reading_dataset, 13, 16)#len(reading_dataset['instances'])-1)


    print("Shortening dataset done!")

    reader = readers[args.model](training_dataset, **vars(args))  # should this be reading_dataset ?

    checkpoint()

    print("Starting to train!")
    train_reader(reader, training_dataset, testing_dataset, args.epochs, args.batch_size,
                 use_train_generator_for_test=True)


if __name__ == "__main__":
    main()
