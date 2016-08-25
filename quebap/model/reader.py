import argparse
import copy
import json
import random

import tensorflow as tf
import numpy as np

from quebap.projects.modelF.structs import FrozenIdentifier
from abc import *


class Batcher(metaclass=ABCMeta):
    """
    A batcher is in charge of converting a reading dataset into batches of tensor flow feed_dicts, and batches
    of tensor values back to reading datasets. A batcher for a MultipleChoiceReader maintains four placeholders:
    * candidates, to represent answer candidates. The number of candidates is determined by the batcher. It can
    choose all candidates for each question, or sub-sample as needed (for example during training).
    * questions, to represent questions
    * target_values, to represent assignments of each candidate to 0/1 truth values.
    * support, to represent a collection of support documents.

    The batcher can represent candidates, support, target_values and questions almost in any way it likes.
    A few exceptions:
    * all placeholder shapes start with [batch_size, ...]
    * target_values need to be a [batch_size, num_candidates] float matrix, where num_candidates is determined by
    the batcher.
    * candidate representation shapes should start with  [batch_size, num_candidates, ...]
    """

    def __init__(self, candidates, questions, target_values, support):
        """
        Create the batcher.
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
        :return: a generator of feed dicts for the batchers placeholders.
        """
        return {}


class MultipleChoiceReader:
    """
    A MultipleChoiceReader reads and answers quebaps with multiple choice questions.
    It provides the interface between quebap files and tensorflow execution
    and optimisation: a batcher that converts quebaps into batched feed_dicts, a scoring TF node over
    answer candidates, and a training loss TF node.
    """

    def __init__(self, batcher: Batcher, scores, loss):
        """

        :param batcher: batcher with a create_batches function (see AtomicBatcher)
        :param scores: [batch_size, num_candidates] TF matrix mapping each candidate in each batch to score
        :param loss: [batch_size] TF vector of per instance losses.
        """
        self.loss = loss
        self.scores = scores
        self.batcher = batcher


def pad_seq(seq, target_length, pad):
    return seq + [pad for _ in range(0, target_length - len(seq))]



class SequenceBatcher(Batcher):
    """
    Converts reading instances into tensors of integer sequences representing tokens. A question batch
    is tranformed into a [batch_size, max_length] integer matrix (question placeholder),
    a list of candidates into a [batch_size, num_candidates, max_length] integer tensor (candidates placeholder)
    the answers are a 0/1 float [batch_size, num_candidates] matrix indicating a true (1) or false (0) label
    for each candidate. (target_values placeholder)
    The difference with respect to the SequenceBatcher is that question lengths are included, for use with the
    Tensorflow dynamic_rnn
    """

    def __init__(self, reference_data, candidate_split=",", question_split="-", support_split=" "):
        """
        Create a new SequenceBatcher.
        :param reference_data: the training data that determines the lexicon.
        :param candidate_split: the regular expression used for tokenizing candidates.
        :param question_split: the regular expression used for tokenizing questions.
        :param support_split: the regular expression used for tokenizing support documents.
        """
        self.reference_data = reference_data
        self.pad = "<PAD>"
        self.candidate_split = candidate_split
        self.question_split = question_split
        self.support_split = support_split

        self.question_lengths = tf.placeholder(tf.int32, (None), name="question_lengths")  # [question_lengths]

        questions = tf.placeholder(tf.int32, (None, None), name="question")  # [batch_size, num_tokens]
        candidates = tf.placeholder(tf.int32, (None, None, None),
                                    name="candidates")  # [batch_size, num_candidates, num_tokens]
        target_values = tf.placeholder(tf.float32, (None, None), name="target")
        support = tf.placeholder(tf.float32, (None, None, None), name="support")

        super().__init__(candidates, questions, target_values, support)

        global_candidates = reference_data['globals']['candidates']
        self.all_candidate_tokens = [self.pad] + sorted({token
                                                         for c in global_candidates
                                                         for token in
                                                         self.string_to_seq(c['text'], candidate_split)})
        instances = reference_data['instances']
        self.all_question_tokens = [self.pad] + sorted({token
                                                        for inst in instances
                                                        for token in
                                                        self.string_to_seq(inst['questions'][0]['question'],
                                                                           question_split)})

        self.all_support_tokens = [self.pad] + sorted({token
                                                       for inst in instances
                                                       for support in inst['support']
                                                       for token in
                                                       self.string_to_seq(support['text'],
                                                                          self.candidate_split)})

        self.question_lexicon = FrozenIdentifier(self.all_question_tokens)
        self.candidate_lexicon = FrozenIdentifier(self.all_candidate_tokens)
        self.support_lexicon = FrozenIdentifier(self.all_support_tokens)

        self.num_candidate_symbols = len(self.candidate_lexicon)
        self.num_questions_symbols = len(self.question_lexicon)
        self.num_support_symbols = len(self.support_lexicon)
        self.max_candidate_length = max([len(self.string_to_seq(c['text'], self.candidate_split))
                                         for c in global_candidates])
        self.global_candidate_seqs = [self.pad_seq([self.candidate_lexicon[t]
                                                    for t in self.string_to_seq(c['text'], self.candidate_split)],
                                                   self.max_candidate_length)
                                      for c in global_candidates]
        self.random = random.Random(0)

    def string_to_seq(self, seq, split, max_length=None):
        result = seq.split(split)
        return result if max_length is None else result + [self.pad for _ in
                                                           range(0, max_length - len(result))]

    def pad_seq(self, seq, target_length):
        return pad_seq(seq, target_length, self.pad)

    def convert_to_predictions(self, candidates, scores):
        """
        Convert a batched candidate tensor and batched scores back into a python dictionary in quebap format.
        :param candidates: candidate representation as generated by this batcher.
        :param scores: scores tensor of the shape of the target_value placeholder.
        :return: sequence of reading instances corresponding to the input.
        """
        all_results = []
        for scores_per_question, candidates_per_question in zip(scores, candidates):
            result_for_question = []
            for score, candidate_seq in zip(scores_per_question, candidates_per_question):
                candidate_tokens = [self.candidate_lexicon.key_by_id(sym) for sym in candidate_seq if
                                    sym != self.candidate_lexicon[self.pad]]
                candidate_text = self.candidate_split.join(candidate_tokens)
                candidate = {
                    'text': candidate_text,
                    'score': score
                }
                result_for_question.append(candidate)
            question = {'answers': sorted(result_for_question, key=lambda x: -x['score'])}
            quebap = {'questions': [question]}
            all_results.append(quebap)
        return all_results

    def create_batches(self, data=None, batch_size=1, test=False):
        """
        Take a dataset and create a generator of (batched) feed_dict objects. At training time this
        batcher sub-samples the candidates (currently one positive and one negative candidate).
        :param data: the input dataset.
        :param batch_size: size of each batch.
        :param test: should this be generated for test time? If so, the candidates are all possible candidates.
        :return: a generator of batches.
        """
        instances = self.reference_data['instances'] if data is None else data['instances']
        for b in range(0, len(instances) // batch_size):
            batch = instances[b * batch_size: (b + 1) * batch_size]

            question_seqs = [[self.question_lexicon[t] for t in
                              self.string_to_seq(inst['questions'][0]['question'], self.question_split)]
                             for inst in batch]
            answer_seqs = [[self.candidate_lexicon[t] for t in
                            self.string_to_seq(inst['questions'][0]['answers'][0]['text'], self.candidate_split)]
                           for inst in batch]
            support_seqs = [[[self.support_lexicon[t]
                              for t in self.string_to_seq(support['text'], self.support_split)]
                             for support in inst['support']]
                            for inst in batch]

            question_length = tf.placeholder(tf.int32, (None), name="question_length")
            question_length = [len(q) for q in question_seqs]

            max_question_length = max([len(q) for q in question_seqs])
            max_answer_length = max([len(a) for a in answer_seqs])
            # we ensure that the number of elements in support, and the number of support documents is at least 1
            # this ensures that in case of empty support we get a single [<EMPTY>] support set that supports treating
            # support uniformly downstream.
            max_support_length = max([len(a) for support in support_seqs for a in support] + [1])
            max_num_support = max([len(support) for support in support_seqs] + [1])

            empty_support = pad_seq([], max_support_length, self.support_lexicon[self.pad])
            answer_seqs_padded = [self.pad_seq(batch_item, max_answer_length) for batch_item in answer_seqs]
            question_seqs_padded = [self.pad_seq(batch_item, max_question_length) for batch_item in question_seqs]
            # [batch_size, max_num_support, max_support_length]
            support_seqs_padded = [
                pad_seq([self.pad_seq(s, max_support_length) for s in batch_item], max_num_support, empty_support)
                for batch_item in support_seqs]

            # sample negative candidate
            if test:
                yield {
                    self.questions: question_seqs_padded,
                    self.question_lengths: question_length,
                    self.candidates: answer_seqs_padded,
                    self.support: support_seqs_padded
                }
            else:
                neg_candidates = [self.random.choice(answer_seqs_padded) for _ in range(0, batch_size)]
                # todo: should go over all questions for same support
                yield {
                    self.questions: question_seqs_padded,
                    self.question_lengths: question_length,
                    self.candidates: [(pos, neg) for pos, neg in zip(answer_seqs_padded, neg_candidates)],
                    self.target_values: [(1.0, 0.0) for _ in range(0, batch_size)],
                    self.support: support_seqs_padded
                }


class AtomicBatcher(Batcher):
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
        self.empty = "<EMPTY>"
        global_candidates = reference_data['globals']['candidates']
        all_candidates = set([c['text'] for c in global_candidates])
        instances = reference_data['instances']
        all_questions = set([inst['questions'][0]['question'] for inst in instances])
        all_support = set([support['text'] for inst in instances for support in inst['support']] + [self.empty])
        self.question_lexicon = FrozenIdentifier(all_questions)
        self.candidate_lexicon = FrozenIdentifier(all_candidates)
        self.support_lexicon = FrozenIdentifier(all_support)

        questions = tf.placeholder(tf.int32, (None,), name='questions')
        candidates = tf.placeholder(tf.int32, (None, None), name="candidates")
        target_values = tf.placeholder(tf.float32, (None, None), name="target_values")
        support = tf.placeholder(tf.float32, (None, None), name="support")
        super().__init__(candidates, questions, target_values, support)

        self.random = random.Random(0)
        self.num_candidates = len(self.candidate_lexicon)
        self.num_questions = len(self.question_lexicon)
        self.num_support = len(self.support_lexicon)

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

            support_ids = [[self.support_lexicon[support['text']] for support in inst['support']]
                           for inst in batch]

            max_num_support = max([len(batch_element) for batch_element in support_ids])

            support_ids_padded = [pad_seq(batch_element, max_num_support, self.empty) for batch_element in support_ids]

            # sample negative candidate
            if test:
                yield {
                    self.questions: question_ids,
                    self.candidates: [list(range(0, self.num_candidates))] * batch_size,
                    self.support: support_ids_padded
                }
            else:
                neg = [self.random.randint(0, len(self.candidate_lexicon) - 1) for _ in range(0, batch_size)]
                # todo: should go over all questions for same support
                yield {
                    self.questions: question_ids,
                    self.candidates: [(pos, neg) for pos, neg in zip(answer_ids, neg)],
                    self.target_values: [(1.0, 0.0) for _ in range(0, batch_size)],
                    self.support: support_ids_padded
                }

    def convert_to_predictions(self, candidates, scores):
        """
        Converts scores and candidate ideas to a prediction output
        :param candidates: [batch_size, num_candidates] int matrix of candidates
        :param scores: [batch_size, num_candidates] float matrix of scores for each candidate
        :return: a list of scored candidate lists consistent with output_schema.json
        """
        all_results = []
        for scores_per_question, candidates_per_question in zip(scores, candidates):
            result_for_question = []
            for score, candidate_id in zip(scores_per_question, candidates_per_question):
                candidate_text = self.candidate_lexicon.key_by_id(candidate_id)
                candidate = {
                    'text': candidate_text,
                    'score': score
                }
                result_for_question.append(candidate)
            question = {'answers': sorted(result_for_question, key=lambda x: -x['score'])}
            quebap = {'questions': [question]}
            all_results.append(quebap)
        return all_results


def count_features(instances, candidates):
    """
    This function computes occurrence count features for all candidates across
    the support given in an instance.
    :param instances: List of quebap instances.
    :param candidates: List of candidate strings to compute count features for.
    :return feature_values: A [n_instances, n_features] float matrix with the computed feature values.
    """
    feature_values = np.zeros([len(instances), len(candidates)], dtype=float)
    for i_inst, inst in enumerate(instances):
        supporting_documents = [support['text'] for support in inst['support']]
        for i_cand, candidate in enumerate(candidates):
            count = 0.0
            for doc in supporting_documents:
                count += doc.count(candidate)
            feature_values[i_inst, i_cand] = count
    return feature_values


class Feature_Batcher(Batcher):
    """
    This batcher wraps quebaps into placeholders, computing features as well:
    1. question_ids: A [batch_size] int vector where each component represents a single question using a single symbol.
    2. candidate_ids: A [batch_size, num_candidates] int matrix where each component represents a candidate answer using
    a single label.
    3. target_values: A [batch_size, num_candidates] float matrix representing the truth state of each candidate using
    1/0 values.
    4. feature_values: A [batch_size, num_features] float matrix representing the feature value of each feature.
    """

    # TODO

    def __init__(self, reference_data, feature_calculator):
        """
        Create a new feature batcher.
        :param reference_data: the quebap dataset to use for initialising the question/candidate to id mapping.
        :param feature_calculator: the function used to compute features, see e.g. count_features().
        """
        self.reference_data = reference_data
        self.empty = "<EMPTY>"
        global_candidates = reference_data['globals']['candidates']
        all_candidates = set([c['text'] for c in global_candidates])
        instances = reference_data['instances']
        all_questions = set([inst['questions'][0]['question'] for inst in instances])
        all_support = set([support['text'] for inst in instances for support in inst['support']] + [self.empty])
        self.question_lexicon = FrozenIdentifier(all_questions)
        self.candidate_lexicon = FrozenIdentifier(all_candidates)
        self.support_lexicon = FrozenIdentifier(all_support)

        self.feature_calculator = feature_calculator

        questions = tf.placeholder(tf.int32, (None,), name='questions')
        candidates = tf.placeholder(tf.int32, (None, None), name="candidates")
        target_values = tf.placeholder(tf.float32, (None, None), name="target_values")
        feature_values = tf.placeholder(tf.float32, (None, None), name="feature_values")
        support = tf.placeholder(tf.float32, (None, None), name="support")
        self.feature_values = feature_values
        super().__init__(candidates, questions, target_values, support)

        self.random = random.Random(0)
        self.num_candidates = len(self.candidate_lexicon)
        self.num_questions = len(self.question_lexicon)
        self.num_support = len(self.support_lexicon)

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

            support_ids = [[self.support_lexicon[support['text']] for support in inst['support']]
                           for inst in batch]

            max_num_support = max([len(batch_element) for batch_element in support_ids])

            support_ids_padded = [pad_seq(batch_element, max_num_support, self.empty) for batch_element in support_ids]

            feature_values = self.feature_calculator(instances, self.candidate_lexicon)

            # sample negative candidate
            if test:
                yield {
                    self.questions: question_ids,
                    self.candidates: [list(range(0, self.num_candidates))] * batch_size,
                    self.support: support_ids_padded,
                    self.feature_values: feature_values
                }
            else:
                neg = [self.random.randint(0, len(self.candidate_lexicon) - 1) for _ in range(0, batch_size)]
                # todo: should go over all questions for same support
                yield {
                    self.questions: question_ids,
                    self.candidates: [(pos, neg) for pos, neg in zip(answer_ids, neg)],
                    self.target_values: [(1.0, 0.0) for _ in range(0, batch_size)],
                    self.support: support_ids_padded,
                    self.feature_values: feature_values
                }

    def convert_to_predictions(self, candidates, scores):
        """
        Converts scores and candidate ideas to a prediction output
        :param candidates: [batch_size, num_candidates] int matrix of candidates
        :param scores: [batch_size, num_candidates] float matrix of scores for each candidate
        :return: a list of scored candidate lists consistent with output_schema.json
        """
        all_results = []
        for scores_per_question, candidates_per_question in zip(scores, candidates):
            result_for_question = []
            for score, candidate_id in zip(scores_per_question, candidates_per_question):
                candidate_text = self.candidate_lexicon.key_by_id(candidate_id)
                candidate = {
                    'text': candidate_text,
                    'score': score
                }
                result_for_question.append(candidate)
            question = {'answers': sorted(result_for_question, key=lambda x: -x['score'])}
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


def create_sequence_embedding(inputs, seq_lengths, repr_dim, vocab_size):
    """
    :param inputs: tensor [d1, ... ,dn] of int32 symbols
    :param seq_lengths: [s1, ..., sn] lengths of instances in the batch
    :param repr_dim: dimension of embeddings
    :param vocab_size: number of symbols
    :return: return [batch_size, repr_dim] tensor representation of symbols.
    """
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, repr_dim], -0.1, 0.1),
                                   name="embedding_matrix", trainable=True)
    # [batch_size, max_seq_length, input_size]
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)

    # dummy test to see if the embedding lookup is working
    # Reduce along dimension 1 (`n_input`) to get a single vector (row) per input example
    # embedding_aggregated = tf.reduce_sum(embedded_inputs, [1])

    cell = tf.nn.rnn_cell.LSTMCell(num_units=repr_dim, state_is_tuple=True)
    # returning [batch_size, max_time, cell.output_size]
    outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float32,
        sequence_length=seq_lengths,
        inputs=embedded_inputs)

    # Getting final state out of dynamic rnn
    shape = tf.shape(outputs)  # [batch_size, max_length, out_dim]
    slice_size = shape * [1, 0, 1] + [0, 1, 0]  # [batch_size, 1 , out_dim]
    slice_begin = shape * [0, 1, 0] + [0, -1, 0]  # [1, max_length-1, 1]
    last_expanded = tf.slice(outputs, slice_begin, slice_size)  # [batch_size, 1, out_dim]
    last = tf.squeeze(last_expanded, [1])  # [batch_size, out_dim]

    return last



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


def create_log_linear_reader(reference_data, **options):
    """
    Create a log-linear reader, i.e. with a log-linear combination of text features.
    :param options: 'repr_dim', dimension of representation .
    :return: ModelLogLinear
    """

    batcher = Feature_Batcher(reference_data, eval(options['feature_type']))

    #here: [n_candidates = n_features]
    candidate_weights = create_dense_embedding(batcher.candidates, batcher.num_candidates, batcher.num_candidates)    #[n_candidates, n_features]
    features = batcher.feature_values   #[batchsize, n_features] = [batchsize, n_candidates]

    scores = create_dot_product_scorer(features, candidate_weights)
    loss = create_softmax_loss(scores, batcher.target_values)
    return MultipleChoiceReader(batcher, scores, loss)


def create_model_f_reader(reference_data, **options):
    """
    Create a ModelF reader.
    :param options: 'repr_dim', dimension of representation .
    :param reference_data: the data to determine the question / answer candidate symbols.
    :return: ModelF
    """
    batcher = AtomicBatcher(reference_data)
    question_encoding = create_dense_embedding(batcher.questions, options['repr_dim'], batcher.num_questions)
    candidate_encoding = create_dense_embedding(batcher.candidates, options['repr_dim'], batcher.num_candidates)
    scores = create_dot_product_scorer(question_encoding, candidate_encoding)
    loss = create_softmax_loss(scores, batcher.target_values)
    return MultipleChoiceReader(batcher, scores, loss)


def create_bag_of_embeddings_reader(reference_data, **options):
    """
    A reader that creates sequence representations of the input reading instance, and then
    models each question and candidate as the sum of the embeddings of their tokens.
    :param reference_data: the reference training set that determines the vocabulary.
    :param options: repr_dim, candidate_split (used for tokenizing candidates), question_split
    :return: a MultipleChoiceReader.
    """
    batcher = SequenceBatcher(reference_data,
                              candidate_split=options['candidate_split'],
                              question_split=options['question_split'])

    # get embeddings for each question token
    # [batch_size, max_question_length, repr_dim]
    question_embeddings = create_dense_embedding(batcher.questions, options['repr_dim'], batcher.num_questions_symbols)
    question_encoding = tf.reduce_sum(question_embeddings, 1)  # [batch_size, repr_dim]

    # [batch_size, num_candidates, max_question_length, repr_dim
    candidate_embeddings = create_dense_embedding(batcher.candidates, options['repr_dim'],
                                                  batcher.num_candidate_symbols)
    candidate_encoding = tf.reduce_sum(candidate_embeddings, 2)  # [batch_size, num_candidates, repr_dim]
    scores = create_dot_product_scorer(question_encoding, candidate_encoding)
    loss = create_softmax_loss(scores, batcher.target_values)
    return MultipleChoiceReader(batcher, scores, loss)


def create_sequence_embeddings_reader(reference_data, **options):
    """
    A reader that creates sequence representations of the input reading instance, and then
    models each question as a sequence encoded with an RNN and candidate as the sum of the embeddings of their tokens.
    :param reference_data: the reference training set that determines the vocabulary.
    :param options: repr_dim, candidate_split (used for tokenizing candidates), question_split
    :return: a MultipleChoiceReader.
    """
    batcher = SequenceBatcher(reference_data,
                                     candidate_split=options['candidate_split'],
                                     question_split=options['question_split'])

    # get embeddings for each question token
    # [batch_size, max_question_length, repr_dim]
    # inputs, seq_lengths, repr_dim, vocab_size

    question_encoding = create_sequence_embedding(batcher.questions, batcher.question_lengths, options['repr_dim'],
                                                  batcher.num_questions_symbols)
    # question_encoding = tf.reduce_sum(question_embeddings, 1)  # [batch_size, repr_dim]

    # [batch_size, num_candidates, max_question_length, repr_dim
    candidate_embeddings = create_dense_embedding(batcher.candidates, options['repr_dim'],
                                                  batcher.num_candidate_symbols)
    candidate_encoding = tf.reduce_sum(candidate_embeddings, 2)  # [batch_size, num_candidates, repr_dim]
    scores = create_dot_product_scorer(question_encoding, candidate_encoding)
    loss = create_softmax_loss(scores, batcher.target_values)
    return MultipleChoiceReader(batcher, scores, loss)


def create_support_bag_of_embeddings_reader(reference_data, **options):
    """
    A reader that creates sequence representations of the input reading instance, and then
    models each question and candidate as the sum of the embeddings of their tokens.
    :param reference_data: the reference training set that determines the vocabulary.
    :param options: repr_dim, candidate_split (used for tokenizing candidates), question_split
    :return: a MultipleChoiceReader.
    """
    batcher = SequenceBatcher(reference_data,
                              candidate_split=options['candidate_split'],
                              question_split=options['question_split'],
                              support_split=options['support_split'])

    candidate_dim = options['repr_dim']
    support_dim = options['support_dim']

    # question embeddings: for each symbol a [support_dim, candidate_dim] matrix
    question_embeddings = tf.Variable(tf.random_normal((batcher.num_questions_symbols, support_dim, candidate_dim)))

    # [batch_size, max_question_length, support_dim, candidate_dim]
    question_encoding_raw = tf.gather(question_embeddings, batcher.questions)

    # question encoding should have shape: [batch_size, 1, support_dim, candidate_dim], so reduce and keep
    question_encoding = tf.reduce_sum(question_encoding_raw, 1, keep_dims=True)

    # candidate embeddings: for each symbol a [candidate_dim] vector
    candidate_embeddings = tf.Variable(tf.random_normal((batcher.num_candidate_symbols, candidate_dim)))
    # [batch_size, num_candidates, max_candidate_length, candidate_dim]
    candidate_encoding_raw = tf.gather(candidate_embeddings, batcher.candidates)

    # candidate embeddings should have shape: [batch_size, num_candidates, 1, candidate_dim]
    candidate_encoding = tf.reduce_sum(candidate_encoding_raw, 2, keep_dims=True)

    # each symbol has [support_dim] vector
    support_embeddings = tf.Variable(tf.random_normal((batcher.num_support_symbols, support_dim)))

    # [batch_size, max_support_num, max_support_length, support_dim]
    support_encoding_raw = tf.gather(support_embeddings, batcher.support)

    # support encoding should have shape: [batch_size, 1, support_dim, 1]
    support_encoding = tf.expand_dims(tf.expand_dims(tf.reduce_sum(support_encoding_raw, (1, 2)), 1), 3)

    # scoring with a dot product
    # [batch_size, num_candidates, support_dim, candidate_dim]
    combined = question_encoding * candidate_encoding * support_encoding
    scores = tf.reduce_sum(combined, (2, 3))

    loss = create_softmax_loss(scores, batcher.target_values)
    return MultipleChoiceReader(batcher, scores, loss)


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
    pass

if __name__ == "__main__":
    main()
