from collections import Counter

import cPickle as pickle
import numpy as np

'''This models the vocabulary and token embeddings'''

from spodernet.utils.util import Logger
log = Logger('vocab.py.txt')

class Vocab(object):
    '''Class that manages work/char embeddings'''

    def __init__(self, path, vocab = Counter(), labels = {}):
        '''Constructor.
        Args:
            vocab: Counter object with vocabulary.
        '''
        token2idx = {}
        idx2token = {}
        self.label2idx = {}
        self.idx2label = {}
        for i, item in enumerate(vocab.items()):
            token2idx[item[0]] = i+1
            idx2token[i+1] = item[0]

        for idx in labels:
            self.label2idx[labels[idx]] = idx
            self.idx2label[idx] = labels[idx]

        # out of vocabulary token
        token2idx['OOV'] = int(0)
        idx2token[int(0)] = 'OOV'
        # empty = 0
        token2idx[''] = int(1)
        idx2token[int(1)] = ''

        self.token2idx = token2idx
        self.idx2token = idx2token
        self.path = path
        if len(idx2token.keys()) > 0:
            self.next_idx = int(np.max(idx2token.keys()) + 1)
        else:
            self.next_idx = int(2)

        if len(self.idx2label.keys()) > 0:
            self.next_label_2dx = int(np.max(self.idx2label.keys()) + 1)
        else:
            self.next_label_idx = int(0)

    @property
    def num_token(self):
        return len(self.token2idx)

    @property
    def num_labels(self):
        return len(self.label2idx)

    def add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = self.next_idx
            self.idx2token[self.next_idx] = token
            self.next_idx += 1

    def add_label(self, label):
        if label not in self.label2idx:
            self.label2idx[label] = self.next_label_idx
            self.idx2label[self.next_label_idx] = label
            self.next_label_idx += 1

    def get_idx(self, word):
        '''Gets the idx if it exists, otherwise returns -1.'''
        if word in self.token2idx:
            return self.token2idx[word]
        else:
            return self.token2idx['OOV']

    def get_idx_label(self, label):
        '''Gets the idx of the label'''
        return self.label2idx[label]

    def get_word(self, idx):
        '''Gets the word if it exists, otherwise returns OOV.'''
        if idx in self.idx2token:
            return self.idx2token[idx]
        else:
            return self.idx2token[0]

    def save_to_disk(self, name=''):
        log.info('Saving vocab to: {0}'.format(self.path))
        pickle.dump([self.token2idx, self.idx2token, self.label2idx,
            self.idx2label], open(self.path + name, 'wb'),
                    pickle.HIGHEST_PROTOCOL)

    def load_from_disk(self, name=''):
        log.info('Loading vocab from: {0}'.format(self.path + name))
        self.token2idx, self.idx2token, self.label2idx, self.idx2label = pickle.load(open(self.path))
