# -*- coding: utf-8 -*-

from jack.util.pipelines import pipeline


def preprocess_with_pipeline(data, vocab, target_vocab, test_time=False, negsamples=0,
                             tokenization=True, sepvocab=True):

    corpus = {"support": [], "question": [], "candidates": [], "ids": []}
    if not test_time:
        corpus["answers"] = []
    for i, xy in enumerate(data):
        x, y = (xy, None) if test_time else xy

        corpus["support"] += [x.support]
        corpus['ids'].append(i)
        corpus["question"].append(x.question)
        corpus["candidates"].append(x.atomic_candidates)
        if not test_time:
            assert len(y) == 1
            corpus["answers"].append([y[0].text])

    corpus, train_vocab, answer_vocab, train_candidates_vocab =\
        pipeline(corpus, vocab, target_vocab, sepvocab=sepvocab, test_time=test_time,
                 tokenization=tokenization, cache_fun=True, map_to_target=False, normalize=True,
                 **({'negsamples': negsamples} if not test_time else {}))

    return corpus, train_vocab, answer_vocab, train_candidates_vocab
