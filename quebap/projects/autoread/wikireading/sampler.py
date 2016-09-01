import os
from quebap.projects.autoread.wikireading import *
from quebap.projects.autoread.wikireading.qa import QASetting
import numpy as np
from tensorflow.models.rnn.ptb import reader
import math


class BatchSampler():
    def __init__(self, sess, dir, filenames, batch_size, max_length, max_vocab, max_answer_vocab, vocab, batches_per_epoch=None):
        self.__fns = [os.path.join(dir, fn) for fn in filenames]
        assert self.__fns, \
            "Created sampler with no examples: directory %s , filenames %s" % (dir, filenames)
        self.__queue = tf.train.string_input_producer(self.__fns)
        self.__reader = tf.TFRecordReader()
        _, record_string = self.__reader.read(self.__queue)
        self.__example = tf.parse_single_example(record_string, tfrecord_features)
        self.__sess = sess
        self.__max_vocab = max_vocab
        self.__max_answer_vocab = max_answer_vocab
        self._max_length = max_length
        self.__batch_size = batch_size
        self.unk_id = vocab["<UNK>"]
        self.start_id = vocab["<S>"]
        self.end_id = vocab["</S>"]
        self._batches_per_epoch = batches_per_epoch
        self.num_batches = 0
        self.epoch = 0

    def get_batch(self):
        batch_qas = []
        for i in range(self.__batch_size):
            for k, v in zip(self.__example.keys(), self.__sess.run(list(self.__example.values()))):
                if k == "document_sequence":
                    context = [w if w < self.__max_vocab else self.unk_id for w in v.values[:self._max_length]]
                elif k == "question_sequence":
                    question = [w if w < self.__max_vocab else self.unk_id for w in v.values]
                elif k == "answer_sequence":
                    answers = [w if w < self.__max_vocab else self.unk_id for w in v.values]
                elif k == "answer_breaks":
                    answer_breaks = v.values
            a = []
            j = 0
            for br in answer_breaks:
                a.append([self.start_id] + answers[j:br] + [self.end_id])
                j = br
            a.append([self.start_id] + answers[j:] + [self.end_id])
            batch_qas.append(QASetting(question, a, context))

        self.num_batches += 1
        if self._batches_per_epoch is None:
            completed = self.__sess.run(self.__reader.num_work_units_completed())
            if completed - self.epoch * len(self.__fns) == len(self.__fns):
                self.epoch += 1
        else:
            if self.num_batches % self._batches_per_epoch == 0:
                self.epoch += 1

        return batch_qas

    def close(self):
        self.__queue.close()


class ContextBatchSampler(BatchSampler):


    def __init__(self, sess, dir, filenames, batch_size, max_length, max_vocab, vocab, batches_per_epoch=None,
                 word_freq=dict(), beta=0.5):
        BatchSampler.__init__(self, sess, dir, filenames, batch_size, max_length, max_vocab, max_vocab, vocab,
                              batches_per_epoch=batches_per_epoch)
        self._word_freq = [1.0] * len(vocab)
        for w, freq in word_freq.items():
            self._word_freq[vocab[w]] = max(float(freq), 1.0)
        self._beta = beta

    def get_batch(self):
        batch = BatchSampler.get_batch(self)
        batch_array = np.zeros([len(batch), self._max_length], np.int64)
        batch_lengths = np.zeros([len(batch)], np.int64)
        batch_weights = np.zeros([len(batch), self._max_length])
        for i, qa_setting in enumerate(batch):
            batch_array[i][:len(qa_setting.context)] = qa_setting.context
            normalizer = len(qa_setting.context)/sum(1.0/math.pow(self._word_freq[w], self._beta) for w in qa_setting.context)
            batch_weights[i][:len(qa_setting.context)] = [1.0/math.pow(self._word_freq[w], self._beta) * normalizer for w in qa_setting.context]
            batch_lengths[i] = len(qa_setting.context)
        return batch_array, batch_lengths, batch_weights


class TextBatchSampler:
    def __init__(self, sess, directory, batch_size, max_length, max_vocab, max_answer_vocab, vocab, epoch_batches=None):
        self.__sess = sess
        self.__max_vocab = max_vocab
        self.__max_answer_vocab = max_answer_vocab
        self._max_length = max_length
        self.__batch_size = batch_size
        self.unk_id = vocab["<UNK>"]
        self.start_id = vocab["<S>"]
        self.end_id = vocab["</S>"]
        self._epoch_batches = epoch_batches
        self.num_batches = 0
        self.epoch = 0

        # todo: read from filename
        train_data, valid_data, test_data, vocabulary = reader.ptb_raw_data(directory)

    def get_batch(self):
        batch = None # todo
        batch_array = np.zeros([len(batch), self._max_length])
        batch_lengths = np.zeros([len(batch)], np.int64)

        for i, sequence in enumerate(batch):
            batch_array[i][:len(sequence)] = sequence
            batch_lengths[i] = len(sequence)
        return batch_array, batch_lengths
