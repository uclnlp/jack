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
                     List[float], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Preprocesses a question and (optionally) answers:
    The steps include tokenization, lower-casing, translation to IDs,
    computing the word-in-question feature, computing token offsets,
    truncating supports, and computing answer spans.
    """
    supports = qa_setting.support
    question = qa_setting.question

    question_tokens, question_ids, question_length, question_lemmas, _ = preprocessing.nlp_preprocess(
        question, vocab, lowercase=lowercase, use_spacy=spacy_nlp,
        lemmatize=lemmatize, with_lemmas=with_lemmas, with_tokens_offsets=False)

    preprocessed_supports = [
        preprocessing.nlp_preprocess(
            support, vocab, lowercase=lowercase, use_spacy=spacy_nlp,
            lemmatize=lemmatize, with_lemmas=with_lemmas, with_tokens_offsets=True)
        for support in supports]

    support_tokens, support_ids, support_length, support_lemmas, char_offsets = \
            zip(*preprocessed_supports)

    token_offsets = [(doc_idx, char_offset)
                     for doc_idx, char_offsets in enumerate(char_offsets)
                     for char_offset in char_offsets]

    support_tokens_flat = [t for s in support_tokens for t in s]
    support_lemmas_flat = None
    if with_lemmas:
        support_lemmas_flat = [t for s in support_lemmas for t in s]
    support_ids_flat = [i for ids in support_ids for i in ids]
    support_length = len(support_tokens_flat)

    rng = random.Random(12345)

    word_in_question = []
    if with_lemmas:
        assert support_lemmas_flat is not None
        for lemma in support_lemmas_flat:
            word_in_question.append(float(lemma in question_lemmas and
                                          (not wiq_contentword or (lemma.isalnum() and not lemma.is_stop))))
    else:
        for token in support_tokens_flat:
            word_in_question.append(float(token in question_tokens and (not wiq_contentword or token.isalnum())))

    min_answer = len(support_tokens_flat)
    max_answer = 0

    answer_spans = []
    if with_answers:
        assert isinstance(answers, list)
        for a in answers:
            document_token_offsets = [offset for doc_idx, offset in token_offsets
                                      if doc_idx == a.doc_idx]

            start = 0
            while start < len(document_token_offsets) and document_token_offsets[start] < a.span[0]:
                start += 1

            if start == len(document_token_offsets):
                continue

            end = start
            while end + 1 < len(document_token_offsets) and document_token_offsets[end + 1] < a.span[1]:
                end += 1

            # Convert (start, end) document token indices into indices in the
            # flattened support
            num_tokens_before = sum([len(s) for s in support_tokens[:a.doc_idx]])
            start += num_tokens_before
            end += num_tokens_before

            if (start, end) not in answer_spans:
                answer_spans.append((start, end))
                min_answer = min(min_answer, start)
                max_answer = max(max_answer, end)

    # cut support whenever there is a maximum allowed length and recompute answer spans
    if max_support_length is not None and len(support_tokens_flat) > max_support_length > 0:
        support_length = max_support_length

        # Find new start and end in the flattened support
        if max_answer < max_support_length:
            new_start = 0
            new_end = max_support_length
        else:
            offset = rng.randint(1, 11)
            new_end = max_answer + offset
            new_start = max(0, min(min_answer - offset, new_end - max_support_length))
            while new_end - new_start > max_support_length:
                answer_spans = [(s, e) for s, e in answer_spans if e < (new_end - offset)]
                new_end = max(answer_spans, key=lambda span: span[1])[1] + offset
                new_start = max(0, min(min_answer - offset, new_end - max_support_length))

        # Crop support according to new start and end pointers
        support_tokens_flat = support_tokens_flat[new_start:new_end]
        support_ids_flat = support_ids_flat[new_start:new_end]
        if with_lemmas:
            support_lemmas_flat = support_lemmas_flat[new_start:new_end]
        answer_spans = [(s - new_start, e - new_start) for s, e in answer_spans]
        word_in_question = word_in_question[new_start:new_end]

    return question_tokens, question_ids, question_lemmas, question_length, \
           support_tokens_flat, support_ids_flat, support_lemmas_flat, support_length, \
           word_in_question, token_offsets, answer_spans
