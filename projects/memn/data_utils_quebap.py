# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import json
from nltk import word_tokenize, sent_tokenize


def load_task(train_file, dev_file, test_file):
    '''
    Returns a tuple containing the training and testing data for the task.
    '''
    train_data = get_stories(train_file)
    dev_data = get_stories(dev_file)
    test_data = get_stories(test_file)
    print(train_file, dev_file, test_file)
    return train_data, dev_data, test_data


def get_stories(f):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(json.load(f))


def parse_stories(jsonfile):
    '''
    Parse stories provided in the jtr format
    '''
    data = []
    for inst in jsonfile['instances']:
        #print(inst)
        story = []
        for t in inst["support"]:
            sents = sent_tokenize(t["text"])
            for s in sents:
                story.append(word_tokenize(s))
            #story.extend(sent_tokenize(word_tokenize(t["text"])))
        for qq in inst["questions"]:
            q = word_tokenize(qq["question"].replace("?", ""))
            for qa in qq["answers"]:
                # take the first answer since they all seem to be the same for squad (apart from the differing white space)
                a = word_tokenize(qa["text"])
                break

            data.append((story, q, a))

    return data


def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for inst in data:
        #print(inst)
        story, query, answer = inst
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)
