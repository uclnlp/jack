import tensorflow as tf
from quebap.projects.clozecompose.tensorizer import *
from quebap.model.models import *

def train_reader(reader: MultipleChoiceReader, train_data, test_data, num_epochs, batch_size,
                 optimiser=tf.train.AdamOptimizer(), use_train_generator_for_test=False):
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
    # todo: make sure basic sequence reader works
    # todo: write bicond seq encoder with dynamic rnn
    readers = {
        'se': create_sequence_embeddings_reader
    }

    parser = argparse.ArgumentParser(description='Train and Evaluate a machine reader')
    parser.add_argument('--train', default='../../data/scienceQA/scienceQA.json', type=argparse.FileType('r'), help="Quebap training file")
    parser.add_argument('--test', default='../../data/scienceQA/scienceQA.json', type=argparse.FileType('r'), help="Quebap test file")
    parser.add_argument('--batch_size', default=2, type=int, metavar="B", help="Batch size (suggestion)")
    parser.add_argument('--repr_dim', default=5, type=int, help="Size of the hidden representation")
    parser.add_argument('--support_dim', default=5, type=int, help="Size of the hidden representation for support")
    parser.add_argument('--model', default='se', choices=sorted(readers.keys()), help="Reading model to use")
    parser.add_argument('--epochs', default=1, type=int, help="Number of epochs to train for")


    args = parser.parse_args()

    reading_dataset = json.load(args.train)

    reader = readers[args.model](reading_dataset, **vars(args))

    train_reader(reader, reading_dataset, reading_dataset, args.epochs, args.batch_size,
                 use_train_generator_for_test=True)


if __name__ == "__main__":
    main()
