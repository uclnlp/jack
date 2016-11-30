import json
from pprint import pprint
from time import time, sleep
from os import path

from math import factorial
from quebap.sisyphos.batch import get_feed_dicts
from quebap.sisyphos.vocab import Vocab, NeuralVocab
from quebap.sisyphos.map import tokenize, lower, deep_map, deep_seq_map
from quebap.sisyphos.models import conditional_reader_model
from quebap.sisyphos.train import train
from quebap.sisyphos.prepare_embeddings import load as loads_embeddings
import tensorflow as tf
import numpy as np
import random
from quebap.sisyphos.hooks import SpeedHook, AccuracyHook, LossHook


def load_corpus(name):
    story = []
    order = []

    with open("./quebap/data/StoryCloze/%s_shuffled.tsv" % name, "r") as f:
        for line in f.readlines():
            splits = [x.strip() for x in line.split("\t")]
            current_story = splits[0:5]
            current_order = splits[5:]

            story.append(current_story)
            order.append(permutation_index(current_order))

    return {"story": story, "order": order}


def permutation_index(p):
    result = 0
    for j in range(len(p)):
        k = sum(1 for i in p[j + 1:] if i < p[j])
        result += k * factorial(len(p) - j - 1)
    return result


train_corpus = load_corpus("train")
dev_corpus = load_corpus("dev")
test_corpus = load_corpus("test")

for i in range(3):
    print(train_corpus["story"][i], train_corpus["order"][i])


def pipeline(corpus, vocab=None, target_vocab=None, emb=None, freeze=False):
    vocab = vocab or Vocab(emb=emb)
    target_vocab = target_vocab or Vocab(unk=None)
    if freeze:
        vocab.freeze()
        target_vocab.freeze()

    corpus_tokenized = deep_map(corpus, tokenize, ["story"])
    corpus_lower = deep_seq_map(corpus_tokenized, lower, ["story"])
    corpus_os = deep_seq_map(corpus_lower,
                             lambda xs: ["<SOS>"] + xs + ["<EOS>"], ["story"])
    corpus_ids = deep_map(corpus_os, vocab, ["story"])
    corpus_ids = deep_map(corpus_ids, target_vocab, ["order"])
    corpus_ids = deep_seq_map(corpus_ids,
                              lambda xs: len(xs), keys=["story"],
                              fun_name='lengths', expand=True)
    return corpus_ids, vocab, target_vocab

train_mapped, train_vocab, train_target_vocab = pipeline(train_corpus)
dev_mapped, _, _ = pipeline(dev_corpus, train_vocab, train_target_vocab)
test_mapped, _, _ = pipeline(test_corpus, train_vocab, train_target_vocab)


for i in range(1):
    print(train_mapped["story"][i],
          train_mapped["story_lengths"][i],
          train_mapped["order"][i])

#print(deep_map(train_corpus, tokenize, ["story"])["story"][0])
