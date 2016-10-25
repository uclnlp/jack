import copy
import json
import random
import collections

import tensorflow as tf
from quebap.projects.modelF.structs import FrozenIdentifier
from abc import *
from nltk import word_tokenize, pos_tag, sent_tokenize
from gensim.models import Phrases, word2vec


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


def shorten_candidate_list(reading_dataset, word2vec_model_path="_skip_multi_big_300features_5minwords_5context"):
    """
    Shortens the list of candidates - remove the ones which are unlikely to be of the right answer type
    :param reading_dataset: quebap dataset
    :param begin: first element to keep
    :param end: index of last element to keep + 1
    :return: dataset with instances with shortened candidate sets.
    """

    model = word2vec.Word2Vec.load(word2vec_model_path)
    keytypes = ['dataset', 'author', 'method', 'algorithm', 'task', 'tool', 'description', 'format', 'preprocessing', 'model', 'classifier', 'analysis']

    #for k in keytypes:
    #    print(k, model.similarity(k, "support_vector_machine")) # linear regression, time series

    #words = ['support_vector_machine']
    #for w in words:
    #    for res in model.most_similar(w, topn=3000):
    #        print(w, res)

    result = copy.copy(reading_dataset)
    result['instances'] = reading_dataset['instances']
    #reading_dataset['instances'][begin:end]
    #print("Number reading instances:", len(reading_dataset['instances']))
    for ii, inst in enumerate(reading_dataset['instances']):
        for iq, q in enumerate(inst['questions']):
            #print("before:", len(reading_dataset['instances'][ii]['questions'][iq]['candidates']))
            for ic, c in enumerate(q['candidates']):
                ct = c['text']
                cr = ct.replace(" ", "_")
                if cr in model.vocab:
                    max_sim = 0.0
                    for kt in keytypes:
                        sim = model.similarity(kt, cr)
                        if sim > max_sim:
                            max_sim = sim
                    if max_sim <= 0.49:
                        #if c in reading_dataset['instances'][ii]['questions'][iq]['answers']:
                            #print("Not removed:", c, max_sim)
                        if not c in reading_dataset['instances'][ii]['questions'][iq]['answers']:
                            #print("Useless cand removed:", c['text'], max_sim)
                            reading_dataset['instances'][ii]['questions'][iq]['candidates'].remove(c)
            #print("after:", len(reading_dataset['instances'][ii]['questions'][iq]['candidates']))


    return reading_dataset


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
        self.useSupport = True
        self.reference_data = reference_data
        self.pad = "<pad>"
        self.none = "<none>"  # for NONE answer / neg instances

        self.question_lengths = tf.placeholder(tf.int32, (None, None), name="question_lengths")  # [batch_size, pos/neg]
        self.candidate_lengths = tf.placeholder(tf.int32, (None, None), name="candidate_lengths")  # [batch_size, num_candidates]
        if self.useSupport == True:
            self.support_lengths = tf.placeholder(tf.int32, (None, None), name="support_lengths")  # [batch_size, num_support]
            support = tf.placeholder(tf.int32, (None, None, None), name="support")  # [batch_size, num_supports, num_tokens]
        else:
            support = []

        questions = tf.placeholder(tf.int32, (None, None, None), name="question")  # [batch_size, pos/neg, num_tokens]
        candidates = tf.placeholder(tf.int32, (None, None, None),
                                    name="candidates")  # [batch_size, num_candidates, num_tokens]
        target_values = tf.placeholder(tf.float32, (None, None, None), name="target") # [batch_size, pos/neg, num_candidates]


        super().__init__(candidates, questions, target_values, support)


        instances = reference_data['instances']

        all_question_tokens = [self.pad, self.none] + [token
                                                        for inst in instances
                                                        for question in inst['questions']
                                                        for token in
                                                        word_tokenize(question['question'])]

        all_support_tokens = [self.pad, self.none] + [token
                                                       for inst in instances
                                                       for support in inst['support']
                                                       for token in
                                                       word_tokenize(support['text'])]

        all_candidate_tokens = [self.pad, self.none] + [token
                                                         for inst in instances
                                                         for question in inst['questions']
                                                         for candidate in question['candidates'] + question['answers']
                                                         for token in
                                                         word_tokenize(candidate['text'])]


        count = [[self.pad, -1], [self.none, -1]]
        for c in all_candidate_tokens:
            count.append([c, -1])
        min_l_vocab = len(count)
        count.extend(collections.Counter(all_question_tokens + all_support_tokens).most_common(50000-min_l_vocab))  # 50000

        self.all_tokens = [t[0] for t in count]



        self.lexicon = FrozenIdentifier(self.all_tokens, default_key=self.none)
        self.num_symbols = len(self.lexicon)


        all_question_seqs = [[self.lexicon[t]
                              for t in word_tokenize(inst['questions'][0]['question'])]
                             for inst in instances]

        quest_len = [len(q) for q in all_question_seqs]
        self.all_max_question_length = max(quest_len)
        print("Max question length", self.all_max_question_length)
        print("Average question length", float(sum(quest_len)) / float(len(quest_len)))

        support_lens = [len(inst['support'])
                        for inst in instances]
        print("Max number supports", max(support_lens))
        print("Average number supports", float(sum(support_lens)) / float(len(support_lens)))

        lens_supports = [len(word_tokenize(support['text']))
         for inst in instances
         for support in inst['support']]
        print("Max num support tokens", max(lens_supports))
        print("Average number support tokens", float(sum(lens_supports)) / float(len(lens_supports)))

        self.all_question_seqs_padded = [pad_seq(q, self.all_max_question_length, self.lexicon[self.pad]) for q in all_question_seqs]

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

            if self.useSupport == True:
                support_seqs = [[[self.lexicon[t]
                              for t in word_tokenize(support['text'])]
                             for support in inst['support']]
                            for inst in batch]
                max_support_length = max([len(a) for support in support_seqs for a in support])
                max_num_support = max([len(supports) for supports in support_seqs])
                # [batch_size, max_num_support, max_support_length]
                empty_support = pad_seq([], max_support_length, self.lexicon[self.pad])
                # support_seqs_padded = [self.pad_seq([self.pad_seq(s, max_support_length) for s in supports], max_num_support) for supports in support_seqs]
                support_seqs_padded = [
                    pad_seq([pad_seq(s, max_support_length, self.lexicon[self.pad]) for s in batch_item],
                            max_num_support, empty_support)
                    for batch_item in support_seqs]

                support_length = [[len(c) for c in pad_seq(inst, max_num_support, [])] for inst in support_seqs]


            candidate_seqs = [[[self.lexicon[t]
                             for t in word_tokenize(candidate['text'])]
                            for candidate in inst['questions'][0]['candidates']]
                           for inst in batch]

            answer_seqs = [[[self.lexicon[t]
                              for t in word_tokenize(answ['text'])]
                             for answ in inst['questions'][0]['answers']]
                            for inst in batch]

            question_seqs = [[self.lexicon[t]
                           for t in word_tokenize(inst['questions'][0]['question'])]
                         for inst in batch]

            #max_question_length = max([len(q) for q in question_seqs])
            max_question_length = self.all_max_question_length
            #max_answer_length = max([len(a) for answer in answer_seqs for a in answer])


            max_candidate_length = max([len(a) for cand in candidate_seqs for a in cand])
            max_num_cands = max([len(cands) for cands in candidate_seqs])
            #max_num_answs = max([len(answs) for answs in answer_seqs])

            # [batch_size, max_question_length]
            question_seqs_padded = [pad_seq(q, max_question_length, self.lexicon[self.pad]) for q in question_seqs]

            # [batch_size, max_num_cands, max_candidate_length]
            empty_candidates = pad_seq([], max_candidate_length, self.lexicon[self.pad])
            candidate_seqs_padded = [
                pad_seq([pad_seq(s, max_candidate_length, self.lexicon[self.pad]) for s in batch_item], max_num_cands, empty_candidates)
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


            # to test dimensionalities
            """print(tf.shape(self.questions), tf.shape(question_seqs_padded))
            print(tf.shape(self.question_lengths), tf.shape(question_length))
            print(tf.shape(self.candidates), tf.shape(candidate_seqs_padded))
            print(tf.shape(self.candidate_lengths), tf.shape(candidate_length))
            print(tf.shape(self.support), tf.shape(support_seqs_padded))
            print(tf.shape(self.target_values), tf.shape(target_values_padded))"""


            # target values for test are not supplied, performance at test time is estimated by printing to converting back to quebaps again

            # sample negative candidate
            if self.useSupport == False:
                if test:
                    yield {
                    self.questions: question_seqs_padded,
                    self.question_lengths: question_length,
                    self.candidates: candidate_seqs_padded,  # !!! also fix in main code
                    self.candidate_lengths: candidate_length
                    }
                else:
                    yield {
                    self.questions: [(pos, neg) for pos, neg in zip(question_seqs_padded, neg_question_seqs_padded)],
                    self.question_lengths: [(pos, neg) for pos, neg in zip(question_length, question_length_neg)],
                    self.candidates: candidate_seqs_padded,
                    self.candidate_lengths: candidate_length,
                    self.target_values: [(pos, neg) for pos, neg in zip(target_values_padded, neg_target_values_padded)]  #[(1.0, 0.0) for _ in range(0, batch_size)],
                    }
            else:
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


    def convert_to_predictions(self, batch, scores):
        """
        Convert a batched candidate tensor and batched scores back into a python dictionary in quebap format.
        :param candidates: candidate representation as generated by this tensorizer.
        :param scores: scores tensor of the shape of the target_value placeholder.
        :return: sequence of reading instances corresponding to the input.
        """
        candidates = batch[self.candidates]
        all_results = []
        for scores_per_question, candidates_per_question in zip(scores, candidates):
            result_for_question = []
            for score, candidate_seq in zip(scores_per_question, candidates_per_question):
                candidate_tokens = [self.lexicon.key_by_id(sym) for sym in candidate_seq if
                                    sym != self.lexicon[self.pad]]
                candidate_text = " ".join(candidate_tokens)
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


def accuracy_multi(gold, guess):
    """
    Calculates how often the top predicted answer matches the any gold answer.
    :param gold: quebap dataset with gold answers.
    :param guess: quebap dataset with predicted answers
    :return: accuracy (matches / total number of questions)
    """
    # test whether the top answer is the gold answer
    correct = 0
    total = 0
    for gold_instance, guess_instance in zip(gold['instances'], guess['instances']):
        for gold_question, guess_question in zip(gold_instance['questions'], guess_instance['questions']):
            tops = []
            for g in gold_question['answers']:
                tops.append(" ".join(word_tokenize(g['text'])))  # workaround to make sure the strings are exactly the same
            target = guess_question['answers'][0]['text']
            corr = 0
            if target in tops:
                corr = 1
                correct += 1
            #print(str(corr), target, tops)
            total += 1
    return correct / total


def mrr_at_k(gold, guess, k, print_details=False):
    """
    Calculates the mean reciprical rank up to a rank of k
    :param gold: quebap dataset with gold answers.
    :param guess: quebap dataset with predicted answers
    :return: mrr at k
    """
    correct = 0.0
    total = 0.0
    for gold_instance, guess_instance in zip(gold['instances'], guess['instances']):
        for gold_question, guess_question in zip(gold_instance['questions'], guess_instance['questions']):
            tops = []
            for g in gold_question['answers']:
                tops.append(" ".join(word_tokenize(g['text'])))  # workaround to make sure the strings are exactly the same
            targets = []
            corr = 0.0
            for i, t in enumerate(guess_question['answers']): # this is already ordered
                if i == k:
                    break
                total += 1
                targets.append(t['text'])
                if t['text'] in tops:
                    corr += (1.0 / (i+1))
                    break  # only the highest one counts, otherwise we can end up with a score > 1.0 as there can be multiple answers

            correct += corr
            if print_details == True:
                print(str(corr), targets, tops)

    return correct / total


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


def build_dataset(words, vocabulary_size=50000, min_count=5):
    """
    Build vocabulary, code based on tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    :param words: list of words in corpus
    :param vocabulary_size: max vocabulary size
    :param min_count: min count for words to be considered
    :return: counts, dictionary mapping words to indeces, reverse dictionary
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        if _ >= min_count:# or _ == -1:  # that's UNK only
            dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print("Final vocab size:", len(dictionary))
    return count, dictionary, reverse_dictionary


def transSentToIO(sent, answs):
    enc = [[0.0, 0.0] for s in sent]
    for a in answs:
        a = a['text'].split(" ")
        startind = index(a, sent)
        for i in range(startind, startind + len(a)):
            enc[i][1] = 1.0  # the first one is O, the second one is I
    return enc

def index(subseq, seq):
    i, n, m = -1, len(seq), len(subseq)
    try:
        while True:
            i = seq.index(subseq[0], i + 1, n - m + 1)
            if subseq == seq[i:i + m]:
                return i
    except ValueError:
        return -1


def tensoriserTest():
    with open('../../data/scienceQA/scienceQA_cloze_withcont_2016-10-9_small.json') as data_file:  # scienceQA.json
        data = json.load(data_file)

    tensorizer = SequenceTensorizerTokens2(data) #SequenceTensorizerTokens(data)
    feed_dict = next(tensorizer.create_batches(data, batch_size=2, test=False))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for placeholder in feed_dict:
            print(placeholder)
            try:
                print_tensor_shape_op = tf.Print(placeholder, [tf.shape(placeholder)], "shape: ")
                print(sess.run(print_tensor_shape_op, feed_dict=feed_dict))
                print()
            except ValueError:
                print("ValueError!")
                continue

def main():
    #pass
    tensoriserTest()

if __name__ == "__main__":
    main()
