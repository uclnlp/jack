# -*- coding: utf-8 -*-

from typing import Mapping, List, Any

import tensorflow as tf
from jack.util.map import get_entry_dims
from jack.util.map import tokenize, notokenize, lower, deep_map, deep_seq_map, dynamic_subsample

from jack.util.vocab import Vocab

"""
Here come different flavours of pipelines, tailored towards particular problems.
Placeholders should be made based on the corpus (not within the models as done previously)
"""


def create_placeholders(corpus, types={}):
    """
    creates placeholders for each entry in the corpus (dict or list).
    Default: int64; specify keys (or indices) for which other type is needed in types dictionary; e.g. types={'scores':tf.float32}
    All dimensions are None
    Assumes corpus is dict.
    """
    assert all(t in [tf.int64, tf.float32, tf.float64, tf.int32] for t in types.values()) and all(k in corpus.keys() for k in types.keys()), \
        "Problem with 'types' argument: Please provide correct tf types for keys in the corpus"

    placeholders = {} if isinstance(corpus, dict) else ['' for _ in range(len(corpus))]
    keys = corpus.keys() if isinstance(corpus, dict) else range(len(corpus))
    # TODO: maybe simplify assuming corpus is always dict

    dims = get_entry_dims(corpus)
    for key in keys:
        typ = tf.int64 if key not in types else types[key]
        shape = [None]*dims[key]
        name = key if isinstance(corpus, dict) else None  # no name if list
        placeholders[key] = tf.placeholder(typ, shape, name)  # guaranteed same keys as corpus
    return placeholders

"""
Below:
set of simple functions that aggregate common pipeline functionality,
to be used to create custom pipelines
"""


# TODO: rewrite such that it works for different types of jack files / models
# this is the general jack pipeline
def pipeline(corpus, vocab=None, target_vocab=None, candidate_vocab=None,
             emb=None, freeze=False, normalize=False, tokenization=True, lowercase=True,
             negsamples=0, sepvocab=True, test_time=False, cache_fun=False, map_to_target=True):
    vocab = vocab or Vocab(emb=emb)
    if sepvocab:
        target_vocab = target_vocab or Vocab(unk=None)
        candidate_vocab = candidate_vocab or Vocab(unk=None)
    if freeze:
        vocab.freeze()
        if sepvocab:
            target_vocab.freeze()
            candidate_vocab.freeze()

    if not sepvocab:
        target_vocab = candidate_vocab = vocab

    corpus_tokenized = deep_map(corpus, tokenize if tokenization else notokenize, ['question', 'support'])

    corpus_lower = deep_seq_map(corpus_tokenized, lower, ['question', 'support']) if lowercase else corpus_tokenized

    corpus_os = deep_seq_map(corpus_lower, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ['question', 'support'])\
        if tokenization else corpus_lower

    corpus_ids = deep_map(corpus_os, vocab, ['question', 'support'])
    if not test_time:
        corpus_ids = deep_map(corpus_ids, target_vocab, ['answers'])
    corpus_ids = deep_map(corpus_ids, candidate_vocab, ['candidates'], cache_fun=cache_fun)
    if map_to_target and not test_time:
        def jtr_map_to_targets(xs, cands_name, ans_name):
            """
            Create cand-length vector for each training instance with 1.0s for cands which are the correct answ and 0.0s for cands which are the wrong answ
            #@todo: integrate this function with the one below - the pipeline() method only works with this function
            """
            # `targets` is a list of lists. Each inner list indicated whether candidate
            # i corresponds to the *first* answer.
            xs["targets"] = [[1.0 if xs[ans_name][i][0] == cand else 0.0 for cand in xs[cands_name][i]]
                             for i in range(len(xs[ans_name]))]
            return xs
        corpus_ids = jtr_map_to_targets(corpus_ids, 'candidates', 'answers')
    # TODO: verify!!!! (candidates and answers have been replaced by id's, but
    # if target_vocab differs from candidate_vocab, there is no guarantee that
    # these are the same). Alternative: use functions in pipeline.py

    corpus_ids = deep_seq_map(corpus_ids, lambda xs: len(xs), keys=['question', 'support'], fun_name='lengths', expand=True)
    if negsamples > 0 and not test_time:
        # we want this to be the last thing we do to candidates
        corpus_ids = dynamic_subsample(corpus_ids, 'candidates', 'answers', how_many=negsamples)
    if normalize:
        corpus_ids = deep_map(corpus_ids, vocab._normalize, keys=['question', 'support'])
    return corpus_ids, vocab, target_vocab, candidate_vocab


def transpose_dict_of_lists(dict_of_lists: Mapping[str, list], keys: List[str]) -> List[Mapping[str, Any]]:
    """Takes a dict of lists, and turns it into a list of dicts."""
    return [{key: dict_of_lists[key][i] for key in keys} for i in range(len(dict_of_lists[keys[0]]))]
