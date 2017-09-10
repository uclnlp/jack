import random
import re

from typing import List, Optional, Tuple, Union

from jtr.util.vocab import Vocab

from jtr.data_structures import QASetting, Answer

import numpy as np

__pattern = re.compile('\w+|[^\w\s]')


def tokenize(text):
    return __pattern.findall(text)


def token_to_char_offsets(text, tokenized_text):
    offsets = []
    offset = 0
    for t in tokenized_text:
        offset = text.index(t, offset)
        offsets.append(offset)
        offset += len(t)
    return offsets


def prepare_data(qa_setting: QASetting,
                 answers: Optional[List[Answer]],
                 vocab: Vocab,
                 lowercase: bool = False,
                 with_answers: bool = False,
                 wiq_contentword: bool = False,
                 with_spacy: bool = False,
                 max_support_length: int = -1) \
            -> Tuple[List[str], List[int], int,
                     List[str], List[int], int,
                     List[float], List[int], List[Tuple[int, int]]]:
    """Preprocesses a question and (optionally) answers:
    The steps include tokenization, lower-casing, translation to IDs,
    computing the word-in-question feature, computing token offsets,
    truncating supports, and computing answer spans.
    """

    if with_spacy:
        import spacy
        nlp = spacy.load("en", parser=False)
        thistokenize = lambda t: nlp(t)
    else:
        thistokenize = tokenize

    support = " ".join(qa_setting.support)
    question = qa_setting.question

    if lowercase:
        support = support.lower()
        question = question.lower()

    support_tokens, question_tokens = thistokenize(support), thistokenize(question)

    rng = random.Random(12345)

    word_in_question = []
    for token in support_tokens:
        if with_spacy:
            word_in_question.append(float(any(token.lemma == t2.lemma for t2 in question_tokens) and
                             (not wiq_contentword or (token.orth_.isalnum() and not token.is_stop))))
        else:
            word_in_question.append(float(token in question_tokens and (not wiq_contentword or token.isalnum())))

    if with_spacy:
        token_offsets = [t.idx for t in support_tokens]
        support_tokens = [t.orth_ for t in support_tokens]
        question_tokens = [t.orth_ for t in question_tokens]
    else:
        # char to token offsets
        token_offsets = token_to_char_offsets(support, support_tokens)

    question_length = len(question_tokens)

    min_answer = len(support_tokens)
    max_answer = 0

    answer_spans = []
    if with_answers:

        assert isinstance(answers, list)

        for a in answers:
            start = 0
            while start < len(token_offsets) and token_offsets[start] < a.span[0]:
                start += 1

            if start == len(token_offsets):
                continue

            end = start
            while end + 1 < len(token_offsets) and token_offsets[end + 1] < a.span[1]:
                end += 1
            if (start, end) not in answer_spans:
                answer_spans.append((start, end))
                min_answer = min(min_answer, start)
                max_answer = max(max_answer, end)

    # cut support whenever there is a maximum allowed length and recompute answer spans
    if max_support_length is not None and len(support_tokens) > max_support_length > 0:
        if max_answer < max_support_length:
            support_tokens = support_tokens[:max_support_length]
            word_in_question = word_in_question[:max_support_length]
        else:
            offset = rng.randint(1, 11)
            new_end = max_answer + offset
            new_start = max(0, min(min_answer - offset, new_end - max_support_length))
            while new_end - new_start > max_support_length:
                answer_spans = [(s, e) for s, e in answer_spans if e < (new_end - offset)]
                new_end = max(answer_spans, key=lambda span: span[1])[1] + offset
                new_start = max(0, min(min_answer - offset, new_end - max_support_length))
            support_tokens = support_tokens[new_start:new_end]
            answer_spans = [(s - new_start, e - new_start) for s, e in answer_spans]
            word_in_question = word_in_question[new_start:new_end]

    support_length = len(support_tokens)

    support_ids, question_ids = vocab(support_tokens), vocab(question_tokens)

    return question_tokens, question_ids, question_length, \
           support_tokens, support_ids, support_length, \
           word_in_question, token_offsets, answer_spans


def unique_words_with_chars(q_tokenized, s_tokenized, char_vocab, indices=None, char_limit=20):
    indices = indices or range(len(q_tokenized))

    unique_words_set = dict()
    unique_words = list()
    unique_word_lengths = list()
    question2unique = list()
    support2unique = list()

    for j in indices:
        q2u = list()
        for w in q_tokenized[j]:
            w = w[:char_limit]
            if w not in unique_words_set:
                unique_word_lengths.append(len(w))
                unique_words.append([char_vocab.get(c, 0) for c in w])
                unique_words_set[w] = len(unique_words_set)
            q2u.append(unique_words_set[w])
        question2unique.append(q2u)
        s2u = list()
        for w in s_tokenized[j]:
            w = w[:char_limit]
            if w not in unique_words_set:
                unique_word_lengths.append(len(w))
                unique_words.append([char_vocab.get(c, 0) for c in w])
                unique_words_set[w] = len(unique_words_set)
            s2u.append(unique_words_set[w])
        support2unique.append(s2u)

    return unique_words, unique_word_lengths, question2unique, support2unique


def char_vocab_from_vocab(vocab):
    char_vocab = dict()
    char_vocab["PAD"] = 0
    for i in range(max(vocab.id2sym.keys()) + 1):
        w = vocab.id2sym.get(i)
        if w is not None:
            for c in w:
                if c not in char_vocab:
                    char_vocab[c] = len(char_vocab)
    return char_vocab


def stack_and_pad(values: List[Union[np.ndarray, int, float]], pad = 0) -> np.ndarray:
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

