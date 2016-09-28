import argparse
import tensorflow as tf
from quebap.projects.clozecompose.tensorizer import *
from quebap.projects.clozecompose.models import *
#from quebap.model.models import *

def train_reader(reader: MultipleChoiceReader, train_data, test_data, num_epochs, batch_size,
                 optimiser=tf.train.AdamOptimizer(learning_rate=0.001), use_train_generator_for_test=False):
    """
    Train a reader, and test on test set.
    :param reader: The reader to train
    :param train_data: the quebap training file
    :param test_data: the quebap test file
    :param num_epochs: number of epochs to train
    :param batch_size: size of each batch
    :param optimiser: the optimiser to use
    :return: Nothing
    """
    opt_op = optimiser.minimize(reader.loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for epoch in range(0, num_epochs):
        print("Epoch:", epoch)
        avg_loss = 0
        count = 0
        predictions_tr = []
        for batch in reader.tensorizer.create_batches(train_data, batch_size=batch_size):
            #print(batch)
            _, loss = sess.run((opt_op, reader.loss), feed_dict=batch)
            avg_loss += loss
            count += 1
            #if count % 2 == 0:
            #    print("Avg Loss: {}".format(avg_loss / count))

            scores = sess.run(reader.scores, feed_dict=batch)
            candidates_ids = batch[reader.tensorizer.candidates]
            predictions_tr += reader.tensorizer.convert_to_predictions(candidates_ids, scores)

        print("Train Loss: ", np.sum(avg_loss) / count)
        #print("AccMulti: ", accuracy_multi(test_data, {'instances': predictions_tr}))
        print("Train MRR@5: ", mrr_at_k(train_data, {'instances': predictions_tr}, 5))

        predictions_test = []
        #i = 0
        for batch in reader.tensorizer.create_batches(test_data, test=not use_train_generator_for_test, batch_size=1):
            #print(i)
            #print(batch)
            scores = sess.run(reader.scores, feed_dict=batch)
            candidates_ids = batch[reader.tensorizer.candidates]
            predictions_test += reader.tensorizer.convert_to_predictions(candidates_ids, scores)
            #i += 1

        print("Test MRR@5: ",
              mrr_at_k(test_data, {'instances': predictions_test}, 5, print_details=False))  # was: accuracy, accuracy_multi

    print("Finished training, predictions on test:")
    predictions = []
    for batch in reader.tensorizer.create_batches(test_data, test=not use_train_generator_for_test, batch_size=1):
        scores = sess.run(reader.scores, feed_dict=batch)
        candidates_ids = batch[reader.tensorizer.candidates]
        predictions += reader.tensorizer.convert_to_predictions(candidates_ids, scores)

    print("Test MRR@5: ", mrr_at_k(test_data, {'instances': predictions}, 5, print_details=True))  # was: accuracy, accuracy_multi


def main():
    readers = {
        'se': create_sequence_embeddings_reader,
        'bowv': create_bowv_embeddings_reader,
        'bowv_nosupport': create_bowv_nosupport_embeddings_reader
    }

    parser = argparse.ArgumentParser(description='Train and Evaluate a machine reader')
    parser.add_argument('--trainKBP', default='../../data/scienceQA/scienceQA_kbp_all.json', type=argparse.FileType('r'), help="Quebap training file")
    parser.add_argument('--trainCloze', default='../../data/scienceQA/scienceQA_cloze_shortcontext.json',
                    type=argparse.FileType('r'), help="Quebap training file")
    parser.add_argument('--testSetup', default='clozeOnly', help="clozeOnly, kbpOnly, kbpForTest, clozeForTest, both")
    #parser.add_argument('--test', default='../../data/scienceQA/scienceQA_kbp_all_nosupport.json', type=argparse.FileType('r'), help="Quebap test file")
    parser.add_argument('--batch_size', default=50, type=int, metavar="B", help="Batch size (suggestion)")
    parser.add_argument('--repr_dim', default=100, type=int, help="Size of the hidden representation")
    parser.add_argument('--support_dim', default=100, type=int, help="Size of the hidden representation for support")
    parser.add_argument('--model', default='se', choices=sorted(readers.keys()), help="Reading model to use")
    parser.add_argument('--epochs', default=8, type=int, help="Number of epochs to train for")


    args = parser.parse_args()

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

        for ii, inst in enumerate(reading_dataset_cloze['instances']):
            print(inst['questions'][0]['question'], inst['questions'][0]['answers'], len(inst['support']))

        # reading_dataset = shorten_candidate_list(reading_dataset)

    if args.testSetup == "both":
        training_dataset = shorten_reading_dataset(reading_dataset, 101, len(reading_dataset['instances']) - 1)
        testing_dataset = shorten_reading_dataset(reading_dataset, 0, 100)

        result = copy.copy(training_dataset)
        result['instances'] = training_dataset['instances'] + reading_dataset_cloze['instances']
        training_dataset = result

    elif args.testSetup == "clozeOnly":
        training_dataset = shorten_reading_dataset(reading_dataset_cloze, 101, len(reading_dataset_cloze['instances']) - 1)
        testing_dataset = shorten_reading_dataset(reading_dataset_cloze, 0, 100)


    #training_dataset = shorten_reading_dataset(reading_dataset, 0, 12)
    #testing_dataset = shorten_reading_dataset(reading_dataset, 13, 16)#len(reading_dataset['instances'])-1)


    print("Shortening dataset done!")

    reader = readers[args.model](training_dataset, **vars(args))  # should this be reading_dataset ?

    print("Starting to train!")
    train_reader(reader, training_dataset, testing_dataset, args.epochs, args.batch_size,
                 use_train_generator_for_test=True)


if __name__ == "__main__":
    main()
