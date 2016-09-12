import argparse
import copy
import json
import random

import tensorflow as tf
import numpy as np
import quebap.util.tfutil as tfutil

from quebap.projects.modelF.structs import FrozenIdentifier
from abc import *
from nltk import word_tokenize, pos_tag, sent_tokenize


class Tensorizer(metaclass=ABCMeta):
    """
    A tensorizer is in charge of converting a reading dataset into batches of tensor flow feed_dicts, and batches
    of tensor values back to reading datasets. A tensorizer for a MultipleChoiceReader maintains four placeholders:
    * candidates, to represent answer candidates. The number of candidates is determined by the tensorizer. It can
    choose all candidates for each question, or sub-sample as needed (for example during training).
    * questions, to represent questions
    * target_values, to represent assignments of each candidate to 0/1 truth values.
    * support, to represent a collection of support documents.

    The tensorizer can represent candidates, support, target_values and questions almost in any way it likes.
    A few exceptions:
    * all placeholder shapes start with [batch_size, ...]
    * target_values need to be a [batch_size, num_candidates] float matrix, where num_candidates is determined by
    the tensorizer.
    * candidate representation shapes should start with  [batch_size, num_candidates, ...]
    """

    def __init__(self, candidates, questions, target_values, support):
        """
        Create the tensorizer.
        :param candidates: a placeholder of shape [batch_size, num_candidates, ...]
        :param questions: a placeholder of shape [batch_size, ...] of question representations.
        :param target_values: a float placeholder of shape [batch_size, num_candidates]
        :param support: a placeholder of shape [batch_size, num_support, ...] representing support documents.
        """
        self.support = support
        self.target_values = target_values
        self.questions = questions
        self.candidates = candidates

    @abstractmethod
    def convert_to_predictions(self, candidates, scores):
        """
        Take a candidate tensor and corresponding scores, and convert into a batch of reading instances.
        :param candidates: a tensor representing candidates.
        :param scores: a tensor representing candidate scores.
        :return:
        """
        pass

    @abstractmethod
    def create_batches(self, data, batch_size: int, test: bool):
        """
        Take reading dataset and return a generator of feed_dicts that represent the batched data.
        :param data: Input data in quebap format to convert and batch.
        :param batch_size: How big should the batch be.
        :param test: is this data for testing (True) or training (False).
        :return: a generator of feed dicts for the tensorizers placeholders.
        """
        return {}


class MultipleChoiceReader:
    """
    A MultipleChoiceReader reads and answers quebaps with multiple choice questions.
    It provides the interface between quebap files and tensorflow execution
    and optimisation: a tensorizer that converts quebaps into batched feed_dicts, a scoring TF node over
    answer candidates, and a training loss TF node.
    """

    def __init__(self, tensorizer: Tensorizer, scores, loss):
        """

        :param tensorizer: tensorizer with a create_batches function (see AtomicTensorizer)
        :param scores: [batch_size, num_candidates] TF matrix mapping each candidate in each batch to score
        :param loss: [batch_size] TF vector of per instance losses.
        """
        self.loss = loss
        self.scores = scores
        self.tensorizer = tensorizer


def pad_seq(seq, target_length, pad):
    return seq + [pad for _ in range(0, target_length - len(seq))]


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


class SequenceTensorizer(Tensorizer):
    """
    Converts reading instances into tensors of integer sequences representing tokens. A question batch
    is tranformed into a [batch_size, max_length] integer matrix (question placeholder),
    a list of candidates into a [batch_size, num_candidates, max_length] integer tensor (candidates placeholder)
    the answers are a 0/1 float [batch_size, num_candidates] matrix indicating a true (1) or false (0) label
    for each candidate. (target_values placeholder)
    The difference with respect to the SequenceTensorizer is that question lengths are included, for use with the
    Tensorflow dynamic_rnn
    """

    def __init__(self, reference_data):
        """
        Create a new SequenceTensorizer.
        :param reference_data: the training data that determines the lexicon.
        :param candidate_split: the regular expression used for tokenizing candidates.
        :param question_split: the regular expression used for tokenizing questions.
        :param support_split: the regular expression used for tokenizing support documents.
        """
        self.reference_data = reference_data
        self.pad = "<pad>"
        self.none = "<NONE>"  # for NONE answer / neg instances

        self.question_lengths = tf.placeholder(tf.int32, (None, None), name="question_lengths")  # [pos/neg, batch_size]
        self.candidate_lengths = tf.placeholder(tf.int32, (None, None), name="candidate_lengths")  # [batch_size, num_candidates]
        self.support_lengths = tf.placeholder(tf.int32, (None, None), name="support_lengths")  # [batch_size, num_support]

        questions = tf.placeholder(tf.int32, (None, None, None), name="question")  # [pos/neg, batch_size, num_tokens]
        candidates = tf.placeholder(tf.int32, (None, None, None),
                                    name="candidates")  # [batch_size, num_candidates, num_tokens]
        target_values = tf.placeholder(tf.float32, (None, None, None), name="target") # [pos/neg, batch_size, num_candidates]
        support = tf.placeholder(tf.int32, (None, None, None), name="support")

        super().__init__(candidates, questions, target_values, support)


        instances = reference_data['instances']

        self.all_question_tokens = [self.pad] + sorted({token
                                                        for inst in instances
                                                        for question in inst['questions']
                                                        for token in
                                                        word_tokenize(question['question'])})

        self.all_support_tokens = [self.pad] + sorted({token
                                                       for inst in instances
                                                       for support in inst['support']
                                                       for token in
                                                       word_tokenize(support['text'])})

        self.all_candidate_tokens = [self.pad] + sorted({token
                                                         for inst in instances
                                                         for question in inst['questions']
                                                         for candidate in question['candidates'] + question['answers']
                                                         for token in
                                                         word_tokenize(candidate['text'])})


        self.question_lexicon = FrozenIdentifier(self.all_question_tokens)
        self.candidate_lexicon = FrozenIdentifier(self.all_candidate_tokens)
        self.support_lexicon = FrozenIdentifier(self.all_support_tokens)

        self.num_candidate_symbols = len(self.candidate_lexicon)
        self.num_questions_symbols = len(self.question_lexicon)
        self.num_support_symbols = len(self.support_lexicon)

        all_question_seqs = [[self.question_lexicon[t]
                              for t in word_tokenize(inst['questions'][0]['question'])]
                             for inst in instances]

        self.all_max_question_length = max([len(q) for q in all_question_seqs])

        self.all_question_seqs_padded = [pad_seq(q, self.all_max_question_length, self.question_lexicon[self.pad]) for q in all_question_seqs]

        self.random = random.Random(0)


    def create_batches(self, data=None, batch_size=1, test=False):
        """
        Take a dataset and create a generator of (batched) feed_dict objects. At training time this
        tensorizer sub-samples the candidates (currently one positive and one negative candidate).
        :param data: the input dataset.
        :param batch_size: size of each batch.
        :param test: should this be generated for test time? If so, the candidates are all possible candidates.
        :return: a generator of batches.
        """

        instances = self.reference_data['instances'] if data is None else data['instances']

        for b in range(0, len(instances) // batch_size):
            batch = instances[b * batch_size: (b + 1) * batch_size]

            support_seqs = [[[self.support_lexicon[t]
                              for t in word_tokenize(support['text'])]
                             for support in inst['support']]
                            for inst in batch]

            candidate_seqs = [[[self.candidate_lexicon[t]
                             for t in word_tokenize(candidate['text'])]
                            for candidate in inst['questions'][0]['candidates']]
                           for inst in batch]

            answer_seqs = [[[self.candidate_lexicon[t]
                              for t in word_tokenize(answ['text'])]
                             for answ in inst['questions'][0]['answers']]
                            for inst in batch]

            question_seqs = [[self.question_lexicon[t]
                           for t in word_tokenize(inst['questions'][0]['question'])]
                         for inst in batch]

            #max_question_length = max([len(q) for q in question_seqs])
            max_question_length = self.all_max_question_length
            max_answer_length = max([len(a) for answer in answer_seqs for a in answer])
            max_support_length = max([len(a) for support in support_seqs for a in support])
            max_candidate_length = max([len(a) for cand in candidate_seqs for a in cand])
            max_num_support = max([len(supports) for supports in support_seqs])
            max_num_cands = max([len(cands) for cands in candidate_seqs])
            max_num_answs = max([len(answs) for answs in answer_seqs])

            # [batch_size, max_question_length]
            question_seqs_padded = [pad_seq(q, max_question_length, self.question_lexicon[self.pad]) for q in question_seqs]

            # [batch_size, max_num_support, max_support_length]
            empty_support = pad_seq([], max_support_length, self.support_lexicon[self.pad])
            #support_seqs_padded = [self.pad_seq([self.pad_seq(s, max_support_length) for s in supports], max_num_support) for supports in support_seqs]
            support_seqs_padded = [pad_seq([pad_seq(s, max_support_length, self.support_lexicon[self.pad]) for s in batch_item], max_num_support, empty_support)
                for batch_item in support_seqs]

            # [batch_size, max_num_cands, max_candidate_length]
            empty_candidates = pad_seq([], max_candidate_length, self.candidate_lexicon[self.pad])
            candidate_seqs_padded = [
                pad_seq([pad_seq(s, max_candidate_length, self.candidate_lexicon[self.pad]) for s in batch_item], max_num_cands, empty_candidates)
                for batch_item in candidate_seqs]


            # A [batch_size, num_candidates] float matrix representing the truth state of each candidate using 1 / 0 values
            # rewrite to work with list comprehension
            target_values = []
            for num, i in enumerate(candidate_seqs):
                iv = []
                for s in i:
                    if s in answer_seqs[num]:
                        iv.append(1.0)
                    else:
                        iv.append(0.0)
                target_values.append(iv)


            neg_question_seqs_padded = []
            for qi in question_seqs_padded:
                rq = []
                while rq == [] or rq == qi:
                    rq = self.random.choice(self.all_question_seqs_padded)
                neg_question_seqs_padded.append(rq)


            # target is a candidate-length vector of 0/1s
            target_values_padded = [[c for c in pad_seq(inst, max_num_cands, 0.0)] for inst in target_values]

            neg_target_values_padded = [[0.0 for c in pad_seq(inst, max_num_cands, 0.0)] for inst in target_values] #tf.zeros(tf.shape(target_values_padded), dtype=tf.int32)

            question_length = [len(q) for q in question_seqs]
            #todo: change to actual lengths
            question_length_neg = [len(q) for q in neg_question_seqs_padded]

            # number of local candidates per instance differs, has to be padded
            candidate_length = [[len(c) for c in pad_seq(inst, max_num_cands, [])] for inst in answer_seqs]

            support_length = [[len(c) for c in pad_seq(inst, max_num_support, [])] for inst in support_seqs]

            # to test dimensionalities
            """print(tf.shape(self.questions), tf.shape(question_seqs_padded))
            print(tf.shape(self.question_lengths), tf.shape(question_length))
            print(tf.shape(self.candidates), tf.shape(candidate_seqs_padded))
            print(tf.shape(self.candidate_lengths), tf.shape(candidate_length))
            print(tf.shape(self.support), tf.shape(support_seqs_padded))
            print(tf.shape(self.target_values), tf.shape(target_values_padded))"""


            # target values for test are not supplied, performance at test time is estimated by printing to converting back to quebaps again

            # sample negative candidate
            if test:
                yield {
                    self.questions: question_seqs_padded,
                    self.question_lengths: question_length,
                    self.candidates: candidate_seqs_padded,  # !!! also fix in main code
                    self.candidate_lengths: candidate_length,
                    self.support: support_seqs_padded,
                    self.support_lengths: support_length
                }
            else:
                yield {
                    self.questions: [(pos, neg) for pos, neg in zip(question_seqs_padded, neg_question_seqs_padded)],
                    self.question_lengths: [(pos, neg) for pos, neg in zip(question_length, question_length_neg)],
                    self.candidates: candidate_seqs_padded,
                    self.candidate_lengths: candidate_length,
                    self.target_values: [(pos, neg) for pos, neg in zip(target_values_padded, neg_target_values_padded)],  #[(1.0, 0.0) for _ in range(0, batch_size)],
                    self.support: support_seqs_padded,
                    self.support_lengths: support_length
                }

    #def pad_seq(self, seq, target_length):
    #    return pad_seq(seq, target_length, self.pad)


    def convert_to_predictions(self, candidates, scores):
        """
        Convert a batched candidate tensor and batched scores back into a python dictionary in quebap format.
        :param candidates: candidate representation as generated by this tensorizer.
        :param scores: scores tensor of the shape of the target_value placeholder.
        :return: sequence of reading instances corresponding to the input.
        """
        # todo: update to work with current batcher
        all_results = []
        for scores_per_question, candidates_per_question in zip(scores, candidates):
            result_for_question = []
            for score, candidate_seq in zip(scores_per_question, candidates_per_question):
                candidate_tokens = [self.candidate_lexicon.key_by_id(sym) for sym in candidate_seq if
                                    sym != self.candidate_lexicon[self.pad]]
                candidate_text = " ".join(
                    candidate_tokens)  # won't work, no candidate_split, tokenisation with nltk
                candidate = {
                    'text': candidate_text,
                    'score': score
                }
                result_for_question.append(candidate)
            question = {'answers': sorted(result_for_question, key=lambda x: -x['score'])}
            quebap = {'questions': [question]}
            all_results.append(quebap)
        return all_results





def create_softmax_loss(scores, target_values):
    """

    :param scores: [batch_size, num_candidates] logit scores
    :param target_values: [batch_size, num_candidates] vector of 0/1 target values.
    :return: [batch_size] vector of losses (or single number of total loss).
    """
    return tf.nn.softmax_cross_entropy_with_logits(scores, target_values)



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
            target = guess_question['answers'][0]['text']
            if top == target:
                correct += 1
            total += 1
    return correct / total


def tensoriserTest():
    with open('../../../quebap/data/scienceQA/snippet.json') as data_file:
        data = json.load(data_file)

    tensorizer = SequenceTensorizer(data)
    feed_dict = next(tensorizer.create_batches(data, batch_size=2))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for placeholder in feed_dict:
            print(placeholder)
            print_tensor_shape_op = tf.Print(placeholder, [tf.shape(placeholder)], "shape: ")
            print(sess.run(print_tensor_shape_op, feed_dict=feed_dict))
            print()

def main():
    #pass
    tensoriserTest()

if __name__ == "__main__":
    main()
