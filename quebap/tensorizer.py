import copy
import random
from abc import *

import numpy as np
import tensorflow as tf

from quebap.projects.modelF.structs import FrozenIdentifier


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


class GenericTensorizer(Tensorizer):
    """
    todo
    """

    def __init__(self, reference_data, candidate_split=" ", question_split=" ", support_split=" "):
        """
        todo
        """
        self.reference_data = reference_data
        # self.pad = "<PAD>"
        self.pad = 0
        self.candidate_split = candidate_split
        self.question_split = question_split
        self.support_split = support_split

        # [batch_size, seq_length]
        questions = tf.placeholder(tf.int32, (None, None), name="question")
        # [batch_size, num_candidates, seq_length]
        candidates = tf.placeholder(tf.int32, (None, None, None), name="candidates")
        # [batch_size, num_support, seq_length]
        support = tf.placeholder(tf.int32, (None, None, None), name="support")

        target_values = tf.placeholder(tf.int32, (None, None), name="target")

        # [batch_size]
        self.question_lengths = tf.placeholder(tf.int32, (None), name="question_lengths")
        # [batch_size, num_candidates]
        self.candidate_lengths = tf.placeholder(tf.int32, (None, None), name="candidate_lengths")
        # [batch_size, num_support, indices]
        self.support_indices = tf.placeholder(tf.int32, (None, None), name="support_indices")

        super().__init__(candidates, questions, target_values, support)

        instances = reference_data['instances']

        global_candidates = {}
        self.has_global_candidates = 'globals' in reference_data
        self.has_local_candidates = 'candidates' in instances[0]['questions'][0]

        if self.has_global_candidates:
            global_candidates = reference_data['globals']['candidates']
            self.all_candidate_tokens = \
                [self.pad] + sorted(
                    {token for c in global_candidates
                     for token in self.string_to_seq(c['text'], candidate_split)}
                )
        elif self.has_local_candidates:
            all_candidate_seqs = [token
                                  for inst in instances
                                  for question in inst['questions']
                                  for candidate in question['candidates']
                                  for token in
                                  self.string_to_seq(candidate['text'], self.candidate_split)
                                  ]
            self.all_candidate_tokens = [self.pad] + sorted(set(all_candidate_seqs))
        else:
            # todo: split!
            all_candidate_seqs = [instance["questions"][0]["answers"][0]["text"]
                                  for instance in instances]
            self.all_candidate_tokens = [self.pad] + sorted(set(all_candidate_seqs))

        self.all_question_tokens = \
            [self.pad] + sorted(
                {token for inst in instances
                 for token in
                 self.string_to_seq(inst['questions'][0]['question'],
                                    question_split)})

        self.all_support_tokens = \
            [self.pad] + sorted(
                {token for inst in instances
                 for support in inst['support'] for token in
                 self.string_to_seq(support['text'], self.candidate_split)})

        self.question_lexicon = FrozenIdentifier(self.all_question_tokens)
        self.candidate_lexicon = FrozenIdentifier(self.all_candidate_tokens)
        self.support_lexicon = FrozenIdentifier(self.all_support_tokens)

        self.num_candidate_symbols = len(self.candidate_lexicon)
        self.num_questions_symbols = len(self.question_lexicon)
        self.num_support_symbols = len(self.support_lexicon)

        candidate_lengths = []
        if self.has_global_candidates:
            candidate_lengths = \
                [len(self.string_to_seq(c['text'], self.candidate_split))
                 for c in global_candidates]
        elif self.has_local_candidates:
            all_candidate_seqs = [[[self.candidate_lexicon[t]
                                for t in self.string_to_seq(candidate['text'], self.candidate_split)]
                               for candidate in inst['questions'][0]['candidates']]
                              for inst in instances]
            # then we want the candidates to be defined locally, for each instance
            # [batch_size, num_candidates, candidate_lengths]
            self.candidate_lengths = tf.placeholder(tf.int32, (None, None), name="candidate_lengths")
            candidate_lengths = [[len(c) for c in inst] for inst in all_candidate_seqs]
        else:
            candidate_lengths = \
                [len(self.string_to_seq(c, self.candidate_split))
                 for c in all_candidate_seqs]

        self.max_candidate_length = 0
        if len(candidate_lengths) > 0:
            self.max_candidate_length = max(candidate_lengths)

        if self.has_local_candidates:
            self.max_candidate_length = max([a for cand in candidate_lengths for a in cand])

        self.global_candidate_seqs = \
            [self.pad_seq(
                [self.candidate_lexicon[t]
                 for t in self.string_to_seq(c['text'], self.candidate_split)],
                self.max_candidate_length)
             for c in global_candidates]
        self.random = random.Random(0)

    def string_to_seq(self, seq, split, max_length=None):
        result = seq.split(split)
        return result if max_length is None \
            else result + [self.pad for _ in range(0, max_length - len(result))]

    def pad_seq(self, seq, target_length):
        return pad_seq(seq, target_length, self.pad)

    def convert_to_predictions(self, candidates, scores):
        """
        Convert a batched candidate tensor and batched scores back into a python dictionary in quebap format.
        :param candidates: candidate representation as generated by this tensorizer.
        :param scores: scores tensor of the shape of the target_value placeholder.
        :return: sequence of reading instances corresponding to the input.
        """

        # todo
        pass

    def create_batches(self, data=None, batch_size=1, test=False):
        """
        Take a dataset and create a generator of (batched) feed_dict objects.

        fixme: At training time this tensorizer sub-samples the candidates
          (currently one positive and one negative candidate).

        :param data: the input dataset
        :param batch_size: size of each batch
        :param test: should this be generated for test time?
          If so, the candidates are all possible candidates
        :return: a generator of batches
        """
        instances = self.reference_data['instances'] \
            if data is None else data['instances']

        for b in range(0, len(instances) // batch_size):
            batch = instances[b * batch_size : (b + 1) * batch_size]

            question_seqs = [[self.question_lexicon[t] for t in
                              self.string_to_seq(inst['questions'][0]['question'],
                                                 self.question_split)]
                             for inst in batch]

            if self.has_local_candidates:
                answer_seqs = [[[self.candidate_lexicon[t]
                                for t in self.string_to_seq(candidate['text'], self.candidate_split)  ]
                               for candidate in inst['questions'][0]['candidates']]
                              for inst in batch]
            else:
                answer_seqs = [[self.candidate_lexicon[t] for t in
                                self.string_to_seq(
                                    inst['questions'][0]['answers'][0]['text'],
                                    self.candidate_split)]
                               for inst in instances]

            support_seqs = [[[self.support_lexicon[t]
                              for t in self.string_to_seq(support['text'],
                                                          self.support_split)]
                             for support in inst['support']]
                            for inst in batch]

            max_question_length = max([len(q) for q in question_seqs])

            if self.has_local_candidates:
                max_answer_length = max([len(a) for answer in answer_seqs for a in answer])
            else:
                max_answer_length = max([len(a) for a in answer_seqs])


            # we ensure that the number of elements in support,
            # and the number of support documents is at least 1
            # this ensures that in case of empty support we get a single
            # [<EMPTY>] support set that supports treating
            # support uniformly downstream.

            max_support_length = max(
                [len(a) for support in support_seqs for a in support] + [1])

            max_num_support = max([len(support) for support in support_seqs] + [1])

            empty_support = pad_seq([], max_support_length,
                                    self.support_lexicon[self.pad])

            # fixme: there can be multiple answer_seqs per batch_item!
            # fixed for has_local_candidates
            if self.has_local_candidates:
                #answer_seqs_padded = [self.pad_seq([self.pad_seq(s, max_support_length) for s in supports], max_num_support) for supports in support_seqs]
                max_candidate_length = max([len(a) for cand in answer_seqs for a in cand])
                max_num_cands = max([len(cands) for cands in answer_seqs])

                empty_answers = pad_seq([], max_answer_length,
                                        self.candidate_lexicon[self.pad])
                answer_seqs_padded = [
                    pad_seq([self.pad_seq(s, max_candidate_length) for s in batch_item], max_num_cands, empty_answers)
                    for batch_item in answer_seqs]

            else:
                answer_seqs_padded = [[self.pad_seq(batch_item, max_answer_length)] for batch_item in answer_seqs]

            question_seqs_padded = [self.pad_seq(batch_item, max_question_length)
                                    for batch_item in question_seqs]

            # [batch_size x max_num_support x max_support_length]
            support_seqs_padded = [
                    pad_seq([self.pad_seq(s, max_support_length) for s in batch_item], max_num_support, empty_support)
                for batch_item in support_seqs]

            question_length = [len(q) for q in question_seqs]

            # workaround because candidates are always the same
            # (the global ones) for this tensorizer
            if self.has_global_candidates:
                candidate_length = [[len(c) for c in self.global_candidate_seqs]
                                    for inst in batch]
            elif self.has_local_candidates:
                # number of local candidates per instance differs, has to be padded
                candidate_length = [[len(c) for c in pad_seq(inst, max_num_cands, [])] for inst in answer_seqs]
            else:
                candidate_length = [[1.0]] * batch_size


            if self.has_local_candidates:
                support_indices = [[len(self.string_to_seq(support['text'], self.support_split))
                                    for support in pad_seq(inst['support'], max_num_support, {"text": "", "id": ""})]
                                   for inst in batch]  # [[len(s) for s in inst] for inst in batch]
            else:
                # todo: check if the padding, as above for the local candidates, is needed here too and add if so
                support_indices = [[len(self.string_to_seq(support['text'], self.support_split))
                                for support in inst['support']]
                               for inst in batch] # [[len(s) for s in inst] for inst in batch]

            print(tf.shape(self.questions), tf.shape(question_seqs_padded))
            print(tf.shape(self.question_lengths), tf.shape(question_length))
            print(tf.shape(self.candidates), tf.shape(answer_seqs_padded))
            print(tf.shape(self.candidate_lengths), tf.shape(candidate_length))
            print(tf.shape(self.target_values), tf.shape([[1.0] for _ in range(0, batch_size)]))
            print(tf.shape(self.support), tf.shape(support_seqs_padded))
            print(tf.shape(self.support_indices), tf.shape(support_indices))

            # todo: sample negative candidate
            feed_dict = {
                self.questions: question_seqs_padded,
                self.question_lengths: question_length,
                self.candidates: answer_seqs_padded,
                self.candidate_lengths: candidate_length,
                self.target_values: [[1.0] for _ in range(0, batch_size)],
                self.support: support_seqs_padded,
                self.support_indices: support_indices
            }

            yield feed_dict


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

    def __init__(self, reference_data, candidate_split=",", question_split="-", support_split=" "):
        """
        Create a new SequenceTensorizer.
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

        self.question_lengths = tf.placeholder(tf.int32, (None), name="question_lengths")  # [batch_size]
        self.candidate_lengths = tf.placeholder(tf.int32, (None, None), name="candidate_lengths")  # [batch_size, num_candidates]
        self.support_lengths = tf.placeholder(tf.int32, (None, None), name="support_lengths")  # [batch_size, num_support]

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
        :param candidates: candidate representation as generated by this tensorizer.
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
        tensorizer sub-samples the candidates (currently one positive and one negative candidate).
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

            question_length = [len(q) for q in question_seqs]

            # workaround because candidates are always the same (the global ones) for this tensorizer
            candidate_length = [[len(c) for c in self.global_candidate_seqs] for inst in batch]

            support_length = [[len(s) for s in inst] for inst in batch]

            # sample negative candidate
            if test:
                yield {
                    self.questions: question_seqs_padded,
                    self.question_lengths: question_length,
                    self.candidates: answer_seqs_padded,
                    self.candidate_lengths: candidate_length,
                    self.support: support_seqs_padded,
                    self.support_lengths: support_length
                }
            else:
                neg_candidates = [self.random.choice(answer_seqs_padded) for _ in range(0, batch_size)]
                # todo: should go over all questions for same support
                yield {
                    self.questions: question_seqs_padded,
                    self.question_lengths: question_length,
                    self.candidates: [(pos, neg) for pos, neg in zip(answer_seqs_padded, neg_candidates)],
                    self.candidate_lengths: candidate_length,
                    self.target_values: [(1.0, 0.0) for _ in range(0, batch_size)],
                    self.support: support_seqs_padded,
                    self.support_lengths: support_length
                }


class AtomicTensorizer(Tensorizer):
    """
    This tensorizer wraps quebaps into placeholders:
    1. question_ids: A [batch_size] int vector where each component represents a single question using a single symbol.
    2. candidate_ids: A [batch_size, num_candidates] int matrix where each component represents a candidate answer using
    a single label.
    3. target_values: A [batch_size, num_candidates] float matrix representing the truth state of each candidate using
    1/0 values.
    """

    def __init__(self, reference_data):
        """
        Create a new atomic tensorizer.
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


class Feature_Tensorizer(Tensorizer):
    """
    This tensorizer wraps quebaps into placeholders, computing features as well:
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
        Create a new feature tensorizer.
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
