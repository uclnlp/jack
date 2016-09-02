import os
import random
import numpy as np
import math

class BatchSampler:
    # white-space tokenized documents per line
    def __init__(self, sess, dir, filenames, batch_size, max_length, max_vocab, vocab, batches_per_epoch=None,
                 word_freq=dict(), beta=0.5):
        self.__sess = sess
        self.__max_vocab = max_vocab
        self._max_length = max_length
        self.__batch_size = batch_size
        self.unk_id = vocab["<UNK>"]
        self.start_id = vocab["<S>"]
        self.end_id = vocab["</S>"]
        self.vocab = vocab
        self._batches_per_epoch = batches_per_epoch
        self.num_batches = 0
        self.epoch = 0
        self._rng = random.Random(28739)
        self._fns = filenames
        self._fn_idx = -1
        self._directory = dir
        self._todo = list()
        self._word_freq = [1.0] * max_vocab
        for w, freq in word_freq.items():
            idx = vocab[w]
            if idx < max_vocab:
                self._word_freq[idx] = max(float(freq), 1.0)
        self._beta = beta

    def _read_next_file(self):
        self._fn_idx = (self._fn_idx + 1) % len(self._fns)
        if self._fn_idx == 0:
            self.epoch += 1
            self._rng.shuffle(self._fns)

        with open(os.path.join(self._directory, self._fns[self._fn_idx]), "rb") as f:
            for l in f:
                l = l.decode("utf-8").strip()
                context = [self.vocab.get(w, self.unk_id) for w in l.split()[:self._max_length]]
                for i, w in enumerate(context):
                    if w >= self.__max_vocab:
                        context[i] = self.unk_id
                self._todo.append(context)

    def get_batch(self):
        if len(self._todo) < self.__batch_size:
            self._read_next_file()
        batch_array = np.zeros([self.__batch_size, self._max_length], np.int64)
        batch_lengths = np.zeros([self.__batch_size], np.int64)
        batch_weights = np.zeros([self.__batch_size, self._max_length])
        for i in range(self.__batch_size):
            context = self._todo[i]
            batch_array[i][:len(context)] = context
            normalizer = sum(
                len(context) / math.pow(self._word_freq[w], self._beta) for w in context)
            batch_weights[i][:len(context)] = [1.0 / math.pow(self._word_freq[w], self._beta) * normalizer
                                                          for w in context]
            batch_lengths[i] = len(context)

        return batch_array, batch_lengths, batch_weights