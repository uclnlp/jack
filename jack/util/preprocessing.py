# -*- coding: utf-8 -*-

import re
from typing import Mapping, List, Any, Union, Tuple, Optional

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

from jack.util.vocab import Vocab


def fill_vocab(qa_settings, vocab=None, lowercase=False, lemmatize=False, spacy_nlp=False):
    vocab = vocab or Vocab(unk=None)
    assert not vocab._frozen, 'Filling frozen vocabs does not make a lot fo sense...'
    for qa_setting in qa_settings:
        nlp_preprocess(qa_setting.question, vocab, lowercase, lemmatize, use_spacy=spacy_nlp)
        for s in qa_setting.support:
            nlp_preprocess(s, vocab, lowercase, lemmatize, use_spacy=spacy_nlp)
    return vocab


__pattern = re.compile('\w+|[^\w\s]')


def tokenize(text, pattern=__pattern):
    return __pattern.findall(text)


def token_to_char_offsets(text, tokenized_text):
    offsets = []
    offset = 0
    for t in tokenized_text:
        offset = text.index(t, offset)
        offsets.append(offset)
        offset += len(t)
    return offsets


def nlp_preprocess_all(qa_settings,
                       vocab: Vocab,
                       lowercase: bool = False,
                       lemmatize: bool = False,
                       with_lemmas: bool = False,
                       with_tokens_offsets: bool = False,
                       use_spacy: bool = False):
    assert not vocab._frozen, 'Filling frozen vocabs does not make a lot fo sense...'
    processed_questions = []
    processed_support = []
    for qa_setting in qa_settings:
        processed_questions.append(
            nlp_preprocess(qa_setting.question, vocab, lowercase, lemmatize, use_spacy=use_spacy))
        processed_support.append([])
        for s in qa_settings.support:
            processed_support[-1].append(nlp_preprocess(s, vocab, lowercase, lemmatize, use_spacy=use_spacy))
    return processed_questions, processed_support


__spacy_nlp = None


def spacy_nlp(parser=False, entity=False, matcher=False):
    import spacy
    global __spacy_nlp
    if __spacy_nlp is None:
        __spacy_nlp = spacy.load("en", parser=parser, entity=entity, matcher=matcher)
    return __spacy_nlp


def nlp_preprocess(text: str,
                   vocab: Vocab,
                   lowercase: bool = False,
                   lemmatize: bool = False,
                   with_lemmas: bool = False,
                   with_tokens_offsets: bool = False,
                   use_spacy: bool = False) \
        -> Tuple[List[str], List[int], int, Optional[List[str]],
                 Optional[List[int]]]:
    """Preprocesses a question and support:
    The steps include tokenization, lower-casing. It also includes the computation of token-to-character offsets for
    the support. Lemmatization is supported in 2 ways. If lemmatize is True then the returned tokens are lemmatized
    and the ids correspond to the lemma ids in the vocab. If with_lemmas and not lemmatize then an additional list
    of the lemmatized token in string form is returned.

    Returns:
        tokens, ids, length, lemmas or None, token_offsets or None
    """
    assert not with_lemmas or use_spacy, "enable spacy when using lemmas"
    assert not lemmatize or use_spacy, "enable spacy when using lemmas"

    if use_spacy:
        import spacy
        nlp = spacy_nlp()
        thistokenize = lambda t: nlp(t)
    else:
        thistokenize = tokenize
    if lowercase:
        text = text.lower()
    tokens = thistokenize(text)

    token_offsets = None
    lemmas = None
    if use_spacy:
        if with_lemmas:
            lemmas = [t.lemma_ for t in tokens]
        if with_tokens_offsets:
            token_offsets = [t.idx for t in tokens]
        tokens = [t.lemma for t in tokens] if lemmatize else [t.orth_ for t in tokens]
    else:
        # char to token offsets
        if with_tokens_offsets:
            token_offsets = token_to_char_offsets(text, tokens)

    length = len(tokens)
    ids = vocab(tokens)

    return tokens, ids, length, lemmas, token_offsets


def transpose_dict_of_lists(dict_of_lists: Mapping[str, list], keys: List[str]) \
        -> List[Mapping[str, Any]]:
    """Takes a dict of lists, and turns it into a list of dicts."""

    return [{key: dict_of_lists[key][i] for key in keys}
            for i in range(len(dict_of_lists[keys[0]]))]


def char_vocab_from_vocab(vocab):
    char_vocab = dict()
    char_vocab["PAD"] = 0
    for i in range(len(vocab)):
        w = vocab.get_sym(i)
        if w is not None:
            for c in w:
                if c not in char_vocab:
                    char_vocab[c] = len(char_vocab)
    return char_vocab


def stack_and_pad(values: List[Union[np.ndarray, int, float]], pad=0) -> np.ndarray:
    """Pads a list of numpy arrays so that they have equal dimensions, then stacks them."""
    if isinstance(values[0], int) or isinstance(values[0], float):
        return np.array(values)

    dims = len(values[0].shape)
    max_shape = [max(sizes) for sizes in zip(*[v.shape for v in values])]

    padded_values = []

    for value in values:
        pad_width = [(0, max_shape[i] - value.shape[i])
                     for i in range(dims)]
        padded_value = np.lib.pad(value, pad_width, mode='constant',
                                  constant_values=pad)
        padded_values.append(padded_value)

    return np.stack(padded_values)


def unique_words_with_chars(tokens, char_vocab, char_limit=20):
    vocab = dict()
    rev_vocab = list()
    unique_words = list()
    unique_word_lengths = list()
    token2unique = list()

    for j in range(len(tokens)):
        t2u = list()
        for w in tokens[j]:
            if w not in vocab:
                unique_word_lengths.append(min(char_limit, len(w)))
                unique_words.append([char_vocab.get(c, 0) for c in w[:char_limit]])
                vocab[w] = len(vocab)
                rev_vocab.append(w)
            t2u.append(vocab[w])
        token2unique.append(t2u)

    return unique_words, unique_word_lengths, token2unique, vocab, rev_vocab


def sort_by_tfidf(reference, candidates):
    tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=spacy.en.STOP_WORDS, decode_error='replace')
    try:
        para_features = tfidf.fit_transform(candidates)
        q_features = tfidf.transform([reference])
    except ValueError:
        return [(i, 0.0) for i in range(len(candidates))]

    dists = pairwise_distances(q_features, para_features, "cosine").ravel()
    sorted_ix = np.lexsort((candidates, dists))  # in case of ties, use the earlier paragraph

    return [(i, 1.0 - dists[i]) for i in sorted_ix]
