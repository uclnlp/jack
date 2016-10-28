import argparse
import json

from quebap.model.models_old import create_log_linear_reader, \
    create_model_f_reader, create_bag_of_embeddings_reader, \
    create_sequence_embeddings_reader, create_support_bag_of_embeddings_reader
from quebap.tensorizer import *


def train_reader(reader: MultipleChoiceReader, train_data, test_data, num_epochs, batch_size,
                 optimiser=tf.train.AdamOptimizer(learning_rate=0.0001), use_train_generator_for_test=False):
    """
    Train a reader, and test on test set. Deprecated as of 27 October 2016, will no longer be updated.
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
        avg_loss = 0
        count = 0
        for batch in reader.tensorizer.create_batches(train_data, batch_size=batch_size):
            _, loss = sess.run((opt_op, reader.loss), feed_dict=batch)
            avg_loss += loss
            count += 1
            if count % 1000 == 0:
                print("Avg Loss: {}".format(avg_loss / count))

                # todo: also run dev during training
    predictions = []
    for batch in reader.tensorizer.create_batches(test_data, test=not use_train_generator_for_test):
        scores = sess.run(reader.scores, feed_dict=batch)
        candidates_ids = batch[reader.tensorizer.candidates]
        predictions += reader.tensorizer.convert_to_predictions(candidates_ids, scores)

    print(accuracy(test_data, {'instances': predictions}))


def main():
    """
    Old quebap training script which uses the tensorizer. Deprecated as of 27 October 2016, will no longer be updated.
    :return:
    """
    reader_models = {
        'log_linear': create_log_linear_reader,
        'model_f': create_model_f_reader,
        'boe': create_bag_of_embeddings_reader,
        'boe_support': create_support_bag_of_embeddings_reader,
        'se': create_sequence_embeddings_reader,
    }

    parser = argparse.ArgumentParser(description='Train and Evaluate a machine reader')
    parser.add_argument('--train', default='data/NYT/naacl2013_train.quebap.json', type=argparse.FileType('r'), help="Quebap training file")
    parser.add_argument('--test', default='data/NYT/naacl2013_train.quebap.json', type=argparse.FileType('r'), help="Quebap test file")
    parser.add_argument('--batch_size', default=5, type=int, metavar="B", help="Batch size (suggestion)")
    parser.add_argument('--repr_dim', default=5, type=int, help="Size of the hidden representation")
    parser.add_argument('--support_dim', default=5, type=int, help="Size of the hidden representation for support")
    parser.add_argument('--model', default='model_f', choices=sorted(reader_models.keys()), help="Reading model to use")
    parser.add_argument('--epochs', default=1, type=int, help="Number of epochs to train for")
    parser.add_argument('--train_begin', default=0, metavar='B', type=int, help="Index of first training instance.")
    parser.add_argument('--train_end', default=-1, metavar='E', type=int,
                        help="Index of last training instance plus 1.")
    parser.add_argument('--candidate_split', default="$", type=str, metavar="S",
                        help="Regular Expression for tokenizing candidates")
    parser.add_argument('--question_split', default="-", type=str, metavar="S",
                        help="Regular Expression for tokenizing questions")
    parser.add_argument('--support_split', default="-", type=str, metavar="S",
                        help="Regular Expression for tokenizing support")
    parser.add_argument('--use_train_generator_for_test', default=False, type=bool, metavar="B",
                        help="Should the training candidate generator be used when testing")
    parser.add_argument('--feature_type', default=None, type=str, metavar="F",
                        help="When using features: type of features.")

    args = parser.parse_args()

    reading_dataset = shorten_reading_dataset(json.load(args.train), args.train_begin, args.train_end)

    reader_model = reader_models[args.model](reading_dataset, **vars(args))

    train_reader(reader_model, reading_dataset, reading_dataset, args.epochs, args.batch_size,
                 use_train_generator_for_test=True)


if __name__ == "__main__":
    main()
