from jtr.pipelines import pipeline
from jtr.preprocess.batch import get_batches
from jtr.preprocess.map import numpify
from jtr.preprocess.vocab import Vocab
import re


def preprocess_with_pipeline(data, vocab, test_time=False, negsamples=0,
        tokenization=True):
        corpus = {"support": [], "question": [], "candidates": []}
        if not test_time:
            corpus["answers"] = []
        for xy in data:
            if test_time:
                x = xy
                y = None
            else:
                x, y = xy
            corpus["support"].append(x.support)
            corpus["question"].append(x.question)
            corpus["candidates"].append(x.atomic_candidates)
            assert len(y) == 1
            if not test_time:
                corpus["answers"].append(y[0].text)
        if not test_time:
            corpus, train_vocab, answer_vocab, train_candidates_vocab  = pipeline(corpus, vocab, sepvocab=False,
                                   test_time=test_time,
                                   tokenization=tokenization,
                                   negsamples=negsamples, cache_fun=True,
                                   map_to_target=False)
        else:
            corpus, train_vocab, answer_vocab, train_candidates_vocab = pipeline(corpus, vocab, sepvocab=False,
                                   test_time=test_time,
                                   tokenization=tokenization,
                                   cache_fun=True, map_to_target=False)
        return corpus, train_vocab, answer_vocab, train_candidates_vocab
