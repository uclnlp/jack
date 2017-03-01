import tensorflow as tf

from jtr.preprocess.map import get_entry_dims
from jtr.preprocess.map import tokenize, notokenize, lower, deep_map, deep_seq_map, dynamic_subsample
from jtr.preprocess.vocab import Vocab




"""
Here come different flavours of pipelines, tailored towards particular problems
"""


def simple_pipeline(corpus, vocab=None, candidate_vocab=None, emb=None, negsamples=0):
    """
    TO DO: docstring
    (replaces original pipeline in training_pipeline; new functionality: returns placeholders as well)
    simple scenario: candidate vocab = target vocab
    """

    corpus, vocab = _create_vocab(corpus, ['question', 'support'], vocab=vocab, emb=emb, lowercase=True, tokens=True, add_length=True)
    corpus, candidate_vocab = _create_vocab(corpus, ['candidates'], vocab=candidate_vocab, unk=None)
    candidate_vocab.freeze()  #to be certain: freeze after first call

    corpus, _ = _create_vocab(corpus, ['answers'], vocab=candidate_vocab, unk=None)

    corpus = _map_to_targets(corpus, 'answers', 'candidates', expand=True, fun_name='binary_vector')


    #todo: make compatible with DynamicSubsampledList

    if negsamples > 0:#we want this to be the last thing we do to candidates
        corpus = dynamic_subsample(corpus, 'candidates', 'answers', how_many=negsamples)
    #todo: not tested yet

    return corpus, vocab, candidate_vocab




"""
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

    placeholders = {} if isinstance(corpus, dict) else ['' for i in range(len(corpus))]
    keys = corpus.keys() if isinstance(corpus, dict) else range(len(corpus))
    #todo: maybe simplify assuming corpus is always dict

    dims = get_entry_dims(corpus)
    for key in keys:
        typ = tf.int64 if not key in types else types[key]
        shape = [None]*dims[key]
        name = key if isinstance(corpus, dict) else None # no name if list
        placeholders[key] = tf.placeholder(typ, shape, name) #guaranteed same keys as corpus

    return placeholders





"""
Below:
set of simple functions that aggregate common pipeline functionality,
to be used to create custom pipelines
"""


def _create_vocab(corpus, keys, vocab=None, emb=None, unk=Vocab.DEFAULT_UNK, lowercase=False, tokens=False, add_os=False, add_length=False):
    if isinstance(corpus, dict):
        assert all([key in corpus for key in keys])
    elif isinstance(corpus, list):
        assert all([key in range(len(corpus)) for key in keys])

    vocab = vocab or Vocab(unk=unk, emb=emb)
    #preprocessing
    if lowercase:
        corpus = deep_map(corpus, lambda x: x.lower(), keys)
    if tokens:
        corpus = deep_map(corpus, tokenize, keys)
    if add_os:
        corpus = deep_seq_map(corpus, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ['question', 'support'])

    #replace symbols by ids + fill up vocab
    corpus = deep_map(corpus, vocab, keys)
    if not vocab.frozen:
        #always return normalized id's (watch out, this does not freeze vocab)
        corpus = deep_map(corpus, vocab._normalize, keys=keys)
    #if unk is None (e.g. for fixed candidates labels) and vocab is frozen: unseen id's are None; remove these
    #this can happen if dev/test set has instances with target labels unseen during training
    if vocab.frozen and unk is None:
        corpus = deep_seq_map(corpus, lambda seq: [s for s in seq if not s is None], keys=keys)

    if add_length:
        corpus = deep_seq_map(corpus, lambda xs: len(xs), keys=keys, fun_name='lengths', expand=True)


    return corpus, vocab



#@todo: rewrite such that it works for different types of jtr files / models
# this is the general jtr pipeline
def pipeline(corpus, vocab=None, target_vocab=None, candidate_vocab=None,
             emb=None, freeze=False, normalize=False, tokenization=True, lowercase=True,
             negsamples=0, sepvocab=True, test_time=False, cache_fun=False, map_to_target=True):
    vocab = vocab or Vocab(emb=emb)
    if sepvocab == True:
        target_vocab = target_vocab or Vocab(unk=None)
        candidate_vocab = candidate_vocab or Vocab(unk=None)
    if freeze:
        vocab.freeze()
        if sepvocab == True:
            target_vocab.freeze()
            candidate_vocab.freeze()

    if sepvocab == False:
        target_vocab = candidate_vocab = vocab
    
    if tokenization == True:
        corpus_tokenized = deep_map(corpus, tokenize, ['question', 'support'])
    else:
        corpus_tokenized =deep_map(corpus, notokenize, ['question', 'support'])

    corpus_lower = deep_seq_map(corpus_tokenized, lower, ['question', 'support']) if lowercase else corpus_tokenized

    if tokenization == True:
        corpus_os = deep_seq_map(corpus_lower, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ['question', 'support'])
    else:
        corpus_os = corpus_lower
    corpus_ids = deep_map(corpus_os, vocab, ['question', 'support'])
    if not test_time:
        corpus_ids = deep_map(corpus_ids, target_vocab, ['answers'])
    corpus_ids = deep_map(corpus_ids, candidate_vocab, ['candidates'], cache_fun=cache_fun)
    if map_to_target and not test_time:
        corpus_ids = jtr_map_to_targets(corpus_ids, 'candidates', 'answers')
    #todo: verify!!!! (candidates and answers have been replaced by id's, but if target_vocab differs from candidate_vocab,
    #todo: there is no guarantee that these are the same)
    #todo: alternative: use functions in pipeline.py

    corpus_ids = deep_seq_map(corpus_ids, lambda xs: len(xs), keys=['question', 'support'], fun_name='lengths', expand=True)
    if negsamples > 0 and not test_time:#we want this to be the last thing we do to candidates
            corpus_ids=dynamic_subsample(corpus_ids,'candidates','answers',how_many=negsamples)
    if normalize:
        corpus_ids = deep_map(corpus_ids, vocab._normalize, keys=['question', 'support'])
    return corpus_ids, vocab, target_vocab, candidate_vocab



def jtr_map_to_targets(xs, cands_name, ans_name):
    """
    Create cand-length vector for each training instance with 1.0s for cands which are the correct answ and 0.0s for cands which are the wrong answ
    #@todo: integrate this function with the one below - the pipeline() method only works with this function
    """
    targs = []
    for i in range(len(xs[ans_name])):
        targ = []
        for cand in xs[cands_name][i]:
            if xs[ans_name][i] == cand:
                targ.append(1.0)
            else:
                targ.append(0.0)
        targs.append(targ)
    xs["targets"] = targs
    return xs


def _map_to_targets(xs, answers_key, candidates_key, expand=False, fun_name='binary_vector'):
    """
    Transforms lists of keys in xs[answers_key] into number-of-candidates long binary vectors,
    containing 1's or 0's depending on whether the answers are in the *corresponding* candidates list.
    (Not consistent over whole dataset if candidate lists vary over instances!)

    Replaces xs[answers_key] if expand=False, else creates new key answers_key+'_'+fun_name.

    (assumes xs is dict)
    So far only limited implementation of 1 level of nested lists
    """
    #todo: more general implementation

    def convert2bin(ans,cand):
        assert isinstance(cand,list) or isinstance(cand,tuple), \
            'candidates must be list or tuple'
        if isinstance(ans,list) or isinstance(ans,tuple):
            assert all(type(ai) == type(ci) for ai in ans for ci in cand), \
                'found different types in answers and candidates lists; revise pipeline'
            return [1 if ci in ans else 0 for ci in cand]
        else: #assume scalar or string
            assert all(type(ci) == type(ans) for ci in cand), \
                'found type mismatch between answer and entries in candidates list; revise pipeline'
            return [1 if ans==ci else 0 for ci in cand]

    targs = []
    for answers, cands in zip(xs[answers_key],xs[candidates_key]):
        targs.append(convert2bin(answers, cands))

    if expand:
        key = '%s_%s'%(answers_key, fun_name)
    else:
        key = answers_key
    xs[key] = targs
    return xs
