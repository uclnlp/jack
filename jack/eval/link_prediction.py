# -*- coding: utf-8 -*-

import logging

import numpy as np

from jack.core import QASetting
from jack.io.load import loaders

logger = logging.getLogger(__name__)


def evaluate(reader, dataset, batch_size):
    triples = {tuple(q.question.split()) for q, a in dataset if a[0].text == "True"}
    entity_set = reader.shared_resources.entity_to_index

    conf = reader.shared_resources.config
    all_triples = set()
    if conf.get('train'):
        all_triples.update(tuple(qa.question.split()) for qa, a in loaders[conf['loader']](conf['train'])
                           if a[0].text == "True")
    if conf.get('dev'):
        all_triples.update(tuple(qa.question.split()) for qa, a in loaders[conf['loader']](conf['dev'])
                           if a[0].text == "True")
    if conf.get('test'):
        all_triples.update(tuple(qa.question.split()) for qa, a in loaders[conf['loader']](conf['test'])
                           if a[0].text == "True")

    def scoring_function(triples):
        scores = []
        for i in range(0, len(triples), batch_size):
            batch_qas = [
                QASetting(" ".join(triples[k]))
                for k in range(i, min(len(triples), (i + batch_size)))]
            if batch_qas:
                for a in reader(batch_qas):
                    scores.append(a.score)
        return scores

    ranks, filtered_ranks = compute_ranks(scoring_function, triples, entity_set, all_triples)
    logger.info('========== Unfiltered Results ==========')
    ranking_summary(ranks)
    logger.info('\n\n========== Filtered Results ==========')
    ranking_summary(filtered_ranks)


def compute_ranks(scoring_function, triples, entity_set, true_triples=None):
    subject_ranks, object_ranks = [], []
    subject_ranks_filtered, object_ranks_filtered = [], []

    for s, p, o in triples:
        subject_triples = [(s, p, o)] + [(x, p, o) for x in entity_set if x != s]
        object_triples = [(s, p, o)] + [(s, p, x) for x in entity_set if x != o]

        subject_triple_scores = np.array(scoring_function(subject_triples))
        object_triple_scores = np.array(scoring_function(object_triples))

        subject_rank = 1 + np.argsort(np.argsort(- subject_triple_scores))[0]
        object_rank = 1 + np.argsort(np.argsort(- object_triple_scores))[0]

        subject_ranks.append(subject_rank)
        object_ranks.append(object_rank)

        if true_triples is None:
            true_triples = []

        for idx, triple in enumerate(subject_triples):
            if triple != (s, p, o) and triple in true_triples:
                subject_triple_scores[idx] = - np.inf

        for idx, triple in enumerate(object_triples):
            if triple != (s, p, o) and triple in true_triples:
                object_triple_scores[idx] = - np.inf

        subject_rank_filtered = 1 + np.argsort(np.argsort(- subject_triple_scores))[0]
        object_rank_filtered = 1 + np.argsort(np.argsort(- object_triple_scores))[0]

        subject_ranks_filtered.append(subject_rank_filtered)
        object_ranks_filtered.append(object_rank_filtered)

    return (subject_ranks, object_ranks), (subject_ranks_filtered, object_ranks_filtered)


def ranking_summary(res, n=10, tag=None):
    dres = dict()

    dres['microlmean'] = np.mean(res[0])
    dres['microlmedian'] = np.median(res[0])
    dres['microlhits@n'] = np.mean(np.asarray(res[0]) <= n) * 100
    dres['micrormean'] = np.mean(res[1])
    dres['micrormedian'] = np.median(res[1])
    dres['microrhits@n'] = np.mean(np.asarray(res[1]) <= n) * 100

    resg = res[0] + res[1]

    dres['microgmean'] = np.mean(resg)
    dres['microgmedian'] = np.median(resg)
    dres['microghits@n'] = np.mean(np.asarray(resg) <= n) * 100

    dres['microlmrr'] = np.mean(1. / np.asarray(res[0]))
    dres['micrormrr'] = np.mean(1. / np.asarray(res[1]))
    dres['microgmrr'] = np.mean(1. / np.asarray(resg))

    logger.info('### MICRO (%s):' % tag)
    logger.info('\t-- left   >> mean: %s, median: %s, mrr: %s, hits@%s: %s%%' %
                (round(dres['microlmean'], 5), round(dres['microlmedian'], 5),
                 round(dres['microlmrr'], 3), n, round(dres['microlhits@n'], 3)))
    logger.info('\t-- right  >> mean: %s, median: %s, mrr: %s, hits@%s: %s%%' %
                (round(dres['micrormean'], 5), round(dres['micrormedian'], 5),
                 round(dres['micrormrr'], 3), n, round(dres['microrhits@n'], 3)))
    logger.info('\t-- global >> mean: %s, median: %s, mrr: %s, hits@%s: %s%%' %
                (round(dres['microgmean'], 5), round(dres['microgmedian'], 5),
                 round(dres['microgmrr'], 3), n, round(dres['microghits@n'], 3)))
