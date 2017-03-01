from jtr.pipelines import pipeline
from jtr.preprocess.batch import get_batches
from jtr.preprocess.map import numpify, deep_map
from jtr.preprocess.vocab import Vocab
import re


def preprocess_with_pipeline(data, vocab, target_vocab, test_time=False, negsamples=0,
        tokenization=True, use_single_support=True, sepvocab=True):
    corpus = {"support": [], "question": [], "candidates": [], "ids" : []}
    if not test_time:
        corpus["answers"] = []
    for i, xy in enumerate(data):
        if test_time:
            x = xy
            y = None
        else:
            x, y = xy
        if use_single_support:
            corpus["support"].append((x.support)[0])
        else:
            corpus["support"].append(x.support)
        corpus['ids'].append(i)
        corpus["question"].append(x.question)
        corpus["candidates"].append(x.atomic_candidates)
        assert len(y) == 1
        if not test_time:
            corpus["answers"].append(y[0].text)
    if not test_time:
        corpus, train_vocab, answer_vocab, train_candidates_vocab  = \
        pipeline(corpus, vocab, target_vocab, sepvocab=sepvocab,
                               test_time=test_time,
                               tokenization=tokenization,
                               negsamples=negsamples, cache_fun=True,
                               map_to_target=False,
                               normalize=True)
    else:
        corpus, train_vocab, answer_vocab, train_candidates_vocab = \
        pipeline(corpus, vocab, target_vocab, sepvocab=sepvocab,
                               test_time=test_time,
                               tokenization=tokenization,
                               cache_fun=True, map_to_target=False,
                               normalize=True)
    return corpus, train_vocab, answer_vocab, train_candidates_vocab
