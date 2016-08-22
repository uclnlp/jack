import argparse
import copy
import json
import random

import tensorflow as tf

from quebap.projects.modelF.structs import FrozenIdentifier


class MultipleChoiceReader:
    """
    A MultipleChoiceReader reads and answers quebaps with multiple choice questions.
    It provides the interface between quebap files and tensorflow execution
    and optimisation: a batcher that converts quebaps into batched feed_dicts, a scoring TF node over
    answer candidates, and a training loss TF node.
    """

    def __init__(self, batcher, scores, loss):
        """

        :param batcher: batcher with a create_batches function (see AtomicBatcher)
        :param scores: [batch_size, num_candidates] TF matrix mapping each candidate in each batch to score
        :param loss: [batch_size] TF vector of per instance losses.
        """
        self.loss = loss
        self.scores = scores
        self.batcher = batcher


class SequenceBatcher:
    def __init__(self, reference_data, candidate_split=",", question_split="-"):
        self.reference_data = reference_data
        self.pad = "PAD"
        self.candidate_split = candidate_split
        self.question_split = question_split
        global_candidates = reference_data['globals']['candidates']
        self.all_candidate_tokens = [self.pad] + sorted({token
                                                         for c in global_candidates
                                                         for token in c['text'].split(candidate_split)})
        instances = reference_data['instances']
        self.all_question_tokens = [self.pad] + sorted({token
                                                        for inst in instances
                                                        for token in
                                                        inst['questions'][0]['question'].split(question_split)})

        self.question_lexicon = FrozenIdentifier(self.all_question_tokens)
        self.candidate_lexicon = FrozenIdentifier(self.all_candidate_tokens)

        self.question_seqs = tf.placeholder(tf.int32, (None, None))  # [batch_size, num_tokens]
        self.candidate_seqs = tf.placeholder(tf.int32, (None, None, None))  # [batch_size, num_candidates, num_tokens]
        self.target_values = tf.placeholder(tf.float32, (None, None))
        self.num_candidate_tokens = len(self.candidate_lexicon)
        self.num_questions_tokens = len(self.question_lexicon)
        self.max_candidate_length = max([len(self.string_to_seq(c['text'], self.candidate_split))
                                         for c in global_candidates])
        self.global_candidate_seqs = [self.pad_seq([self.candidate_lexicon[t]
                                                    for t in self.string_to_seq(c['text'], self.candidate_split)],
                                                   self.max_candidate_length)
                                      for c in global_candidates]
        self.random = random.Random(0)

    def string_to_seq(self, seq, split, max_length=None):
        result = seq.split(split)
        return result if max_length is None else result + [self.pad for _ in range(0, max_length - len(result))]

    def pad_seq(self, seq, target_length):
        return seq + [self.pad for _ in range(0, target_length - len(seq))]

    def create_batches(self, data=None, batch_size=1, test=False):
        instances = self.reference_data['instances'] if data is None else data['instances']
        for b in range(0, len(instances) // batch_size):
            batch = instances[b * batch_size: (b + 1) * batch_size]

            question_seqs = [[self.question_lexicon[t] for t in
                              self.string_to_seq(inst['questions'][0]['question'], self.question_split)]
                             for inst in batch]
            answer_seqs = [[self.candidate_lexicon[t] for t in
                            self.string_to_seq(inst['questions'][0]['answers'][0]['text'], self.candidate_split)]
                           for inst in batch]

            max_question_length = max([len(q) for q in question_seqs])
            max_answer_length = max([len(a) for a in answer_seqs])

            answer_seqs_padded = [self.pad_seq(batch_item, max_answer_length) for batch_item in answer_seqs]
            question_seqs_padded = [self.pad_seq(batch_item, max_question_length) for batch_item in question_seqs]

            # sample negative candidate
            if test:
                yield {
                    self.question_seqs: question_seqs_padded,
                    self.candidate_seqs: answer_seqs_padded
                }
            else:
                neg_candidates = [self.random.choice(answer_seqs_padded) for _ in range(0, batch_size)]
                # todo: should go over all questions for same support
                yield {
                    self.question_seqs: question_seqs_padded,
                    self.candidate_seqs: [(pos, neg) for pos, neg in zip(answer_seqs_padded, neg_candidates)],
                    self.target_values: [(1.0, 0.0) for _ in range(0, batch_size)]
                }


class AtomicBatcher:
    """
    This batcher wraps quebaps into placeholders:
    1. question_ids: A [batch_size] int vector where each component represents a single question using a single symbol.
    2. candidate_ids: A [batch_size, num_candidates] int matrix where each component represents a candidate answer using
    a single label.
    3. target_values: A [batch_size, num_candidates] float matrix representing the truth state of each candidate using
    1/0 values.
    """

    def __init__(self, reference_data):
        """
        Create a new atomic batcher.
        :param reference_data: the quebap dataset to use for initialising the question/candidate to id mapping.
        """
        self.reference_data = reference_data
        global_candidates = reference_data['globals']['candidates']
        all_candidates = set([c['text'] for c in global_candidates])
        instances = reference_data['instances']
        all_questions = set([inst['questions'][0]['question'] for inst in instances])
        self.question_lexicon = FrozenIdentifier(all_questions)
        self.candidate_lexicon = FrozenIdentifier(all_candidates)

        self.question_ids = tf.placeholder(tf.int32, (None,))
        self.candidate_ids = tf.placeholder(tf.int32, (None, None))
        self.target_values = tf.placeholder(tf.float32, (None, None))
        self.random = random.Random(0)
        self.num_candidates = len(self.candidate_lexicon)
        self.num_questions = len(self.question_lexicon)

    def create_batches(self, data=None, batch_size=1, test=False):
        """
        Creates a generator of batch feed_dicts. For training sets a single positive answer and a single negative
        answer are sampled for each question in the batch.
        :param data: data to convert into a generator of feed dicts, one per batch.
        :param batch_size: how large should each batch be.
        :param test: is this a training or test set.
        :return: a generator of batched feed_dicts.
        """
        instances = self.reference_data['instances'] if data is None else data['instances']
        for b in range(0, len(instances) // batch_size):
            batch = instances[b * batch_size: (b + 1) * batch_size]
            question_ids = [self.question_lexicon[inst['questions'][0]['question']]
                            for inst in batch]
            answer_ids = [self.candidate_lexicon[inst['questions'][0]['answers'][0]['text']]
                          for inst in batch]

            # sample negative candidate
            if test:
                yield {
                    self.question_ids: question_ids,
                    self.candidate_ids: [list(range(0, self.num_candidates))] * batch_size
                }
            else:
                neg = [self.random.randint(0, len(self.candidate_lexicon) - 1) for _ in range(0, batch_size)]
                # todo: should go over all questions for same support
                yield {
                    self.question_ids: question_ids,
                    self.candidate_ids: [(pos, neg) for pos, neg in zip(answer_ids, neg)],
                    self.target_values: [(1.0, 0.0) for _ in range(0, batch_size)]
                }

    def convert_to_predictions(self, candidate_ids_values, scores_values):
        """
        Converts scores and candidate ideas to a prediction output
        :param candidate_ids_values: [batch_size, num_candidates] int matrix of candidates
        :param scores_values: [batch_size, num_candidates] float matrix of scores for each candidate
        :return: a list of scored candidate lists consistent with output_schema.json
        """
        all_results = []
        for scores_per_question, candidates_per_question in zip(scores_values, candidate_ids_values):
            result_for_question = []
            for score, candidate_id in zip(scores_per_question, candidates_per_question):
                candidate_text = self.candidate_lexicon.key_by_id(candidate_id)
                candidate = {
                    'text': candidate_text,
                    'score': score
                }
                result_for_question.append(candidate)
            question = {'candidates': sorted(result_for_question, key=lambda x: -x['score'])}
            quebap = {'questions': [question]}
            all_results.append(quebap)
        return all_results


def create_dense_embedding(ids, repr_dim, num_symbols):
    """
    :param ids: tensor [d1, ... ,dn] of int32 symbols
    :param repr_dim: dimension of embeddings
    :param num_symbols: number of symbols
    :return: [d1, ... ,dn,repr_dim] tensor representation of symbols.
    """
    embeddings = tf.Variable(tf.random_normal((num_symbols, repr_dim)))
    encodings = tf.gather(embeddings, ids)  # [batch_size, repr_dim]
    return encodings


def create_dot_product_scorer(question_encodings, candidate_encodings):
    """

    :param question_encodings: [batch_size, enc_dim] tensor of question representations
    :param candidate_encodings: [batch_size, num_candidates, enc_dim] tensor of candidate encodings
    :return: a [batch_size, num_candidate] tensor of scores for each candidate
    """
    return tf.reduce_sum(tf.expand_dims(question_encodings, 1) * candidate_encodings, 2)


def create_softmax_loss(scores, target_values):
    """

    :param scores: [batch_size, num_candidates] logit scores
    :param target_values: [batch_size, num_candidates] vector of 0/1 target values.
    :return: [batch_size] vector of losses (or single number of total loss).
    """
    return tf.nn.softmax_cross_entropy_with_logits(scores, target_values)


def create_model_f_reader(reference_data, **options):
    """
    Create a ModelF reader.
    :param options: 'repr_dim', dimension of representation .
    :param reference_data: the data to determine the question / answer candidate symbols.
    :return: ModelF
    """
    batcher = AtomicBatcher(reference_data)
    question_encoder = create_dense_embedding(batcher.question_ids, options['repr_dim'], batcher.num_questions)
    candidate_encoder = create_dense_embedding(batcher.candidate_ids, options['repr_dim'], batcher.num_candidates)
    scores = create_dot_product_scorer(question_encoder, candidate_encoder)
    loss = create_softmax_loss(scores, batcher.target_values)
    return MultipleChoiceReader(batcher, scores, loss)


def train_reader(reader: MultipleChoiceReader, train_data, test_data, num_epochs, batch_size,
                 optimiser=tf.train.AdamOptimizer()):
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
        for batch in reader.batcher.create_batches(train_data, batch_size=batch_size):
            _, loss = sess.run((opt_op, reader.loss), feed_dict=batch)
            avg_loss += loss
            count += 1
            if count % 1000 == 0:
                print("Avg Loss: {}".format(avg_loss / count))

                # todo: also run dev during training
    predictions = []
    for batch in reader.batcher.create_batches(test_data, test=True):
        scores = sess.run(reader.scores, feed_dict=batch)
        candidates_ids = batch[reader.batcher.candidate_ids]
        predictions += reader.batcher.convert_to_predictions(candidates_ids, scores)

    print(accuracy(test_data, {'instances': predictions}))


def accuracy(gold, guess):
    """
    Calculates how often the top predicted answer matches the first gold answer.
    :param gold: quebap dataset with gold answers.
    :param guess: quebap dataset with predicted answers
    :return: accuracy (matches / total number of questions)
    """
    # test whether the top answer is the gold answer
    correct = 0
    total = 0
    for gold_instance, guess_instance in zip(gold['instances'], guess['instances']):
        for gold_question, guess_question in zip(gold_instance['questions'], guess_instance['questions']):
            top = gold_question['answers'][0]['text']
            target = guess_question['candidates'][0]['text']
            if top == target:
                correct += 1
            total += 1
    return correct / total


def shorten_reading_dataset(reading_dataset, begin, end):
    """
    Shortens the instances list of the dataset, keeping all meta information intact.
    :param reading_dataset: quebap dataset
    :param begin: first element to keep
    :param end: index of last element to keep + 1
    :return: dataset with shortened instances.
    """
    result = copy.copy(reading_dataset)
    result['instances'] = reading_dataset['instances'][begin:end]
    return result


def main():
    readers = {
        'model_f': create_model_f_reader
    }

    parser = argparse.ArgumentParser(description='Train and Evaluate a machine reader')
    parser.add_argument('--train', type=argparse.FileType('r'), help="Quebap training file")
    parser.add_argument('--test', type=argparse.FileType('r'), help="Quebap test file")
    parser.add_argument('--batch_size', default=5, type=int, metavar="B", help="Batch size (suggestion)")
    parser.add_argument('--repr_dim', default=5, type=int, help="Size of the hidden representation")
    parser.add_argument('--model', default='model_f', choices=sorted(readers.keys()), help="Reading model to use")
    parser.add_argument('--epochs', default=1, type=int, help="Number of epochs to train for")
    parser.add_argument('--train_begin', default=0, metavar='B', type=int, help="Index of first training instance.")
    parser.add_argument('--train_end', default=-1, metavar='E', type=int,
                        help="Index of last training instance plus 1.")

    args = parser.parse_args()

    reading_dataset = shorten_reading_dataset(json.load(args.train), args.train_begin, args.train_end)

    reader = readers[args.model](reading_dataset, **vars(args))

    train_reader(reader, reading_dataset, reading_dataset, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
