import random
import re
from typing import List, Optional, Tuple

from jack.core.data_structures import QASetting, Answer
from jack.util import preprocessing
from jack.util.vocab import Vocab

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
                 spacy_nlp: bool = False,
                 max_support_length: int = -1,
                 lemmatize=False,
                 with_lemmas=False) \
        -> Tuple[List[str], List[int], Optional[List[int]], int,
                     List[str], List[int], Optional[List[int]], int,
                     List[float], List[int], List[Tuple[int, int]]]:
    """Preprocesses a question and (optionally) answers:
    The steps include tokenization, lower-casing, translation to IDs,
    computing the word-in-question feature, computing token offsets,
    truncating supports, and computing answer spans.
    """
    support = " ".join(qa_setting.support)
    question = qa_setting.question

    question_tokens, question_ids, question_length, question_lemmas, _ = preprocessing.nlp_preprocess(
        question, vocab, lowercase=lowercase, use_spacy=spacy_nlp,
        lemmatize=lemmatize, with_lemmas=with_lemmas, with_tokens_offsets=False)

    support_tokens, support_ids, support_length, support_lemmas, token_offsets = preprocessing.nlp_preprocess(
        support, vocab, lowercase=lowercase, use_spacy=spacy_nlp,
        lemmatize=lemmatize, with_lemmas=with_lemmas, with_tokens_offsets=True)

    rng = random.Random(12345)

    word_in_question = []

    if with_lemmas:
        assert support_lemmas is not None
        for lemma in support_lemmas:
            word_in_question.append(float(lemma in question_lemmas and
                                          (not wiq_contentword or (lemma.isalnum() and not lemma.is_stop))))
    else:
        for token in support_tokens:
            word_in_question.append(float(token in question_tokens and (not wiq_contentword or token.isalnum())))

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
        support_length = max_support_length
        if max_answer < max_support_length:
            support_tokens = support_tokens[:max_support_length]
            support_ids = support_ids[:max_support_length]
            if with_lemmas:
                support_lemmas = support_lemmas[:max_support_length]
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
            support_ids = support_ids[new_start:new_end]
            if with_lemmas:
                support_lemmas = support_lemmas[new_start:new_end]
            answer_spans = [(s - new_start, e - new_start) for s, e in answer_spans]
            word_in_question = word_in_question[new_start:new_end]

    return question_tokens, question_ids, question_lemmas, question_length, \
           support_tokens, support_ids, support_lemmas, support_length, \
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
