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
                 max_support_length: int = None,
                 lemmatize=False,
                 with_lemmas=False) \
        -> Tuple[List[str], List[int], Optional[List[int]], int,
                 List[List[str]], List[List[int]], Optional[List[List[int]]], List[int],
                 List[List[float]], List[List[int]], List[List[Tuple[int, int]]]]:
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
    question_tokens_set = set(t.lower() for t in question_tokens)

    preprocessed_supports = [
        preprocessing.nlp_preprocess(
            support, vocab, lowercase=lowercase, use_spacy=spacy_nlp,
            lemmatize=lemmatize, with_lemmas=with_lemmas, with_tokens_offsets=True)
        for support in supports]

    all_support_tokens = [s[0] for s in preprocessed_supports]
    all_support_ids = [s[1] for s in preprocessed_supports]
    all_support_length = [s[2] for s in preprocessed_supports]
    all_support_lemmas = [s[3] for s in preprocessed_supports]
    all_token_offsets = [s[4] for s in preprocessed_supports]

    rng = random.Random(12345)

    all_word_in_question = []
    if with_lemmas:
        assert all_support_lemmas is not None
        for support_lemmas in all_support_lemmas:
            all_word_in_question.append([])
            if with_lemmas:
                for lemma in support_lemmas:
                    all_word_in_question[-1].append(float(
                        lemma in question_lemmas and (not wiq_contentword or (lemma.isalnum() and not lemma.is_stop))))
    else:
        for support_tokens in all_support_tokens:
            all_word_in_question.append([])
            for token in support_tokens:
                all_word_in_question[-1].append(
                    float(token.lower() in question_tokens_set and (not wiq_contentword or token.isalnum())))

    all_answer_spans = []
    for doc_idx, support_tokens in enumerate(all_support_tokens):
        min_answer = len(support_tokens)
        max_answer = 0
        token_offsets = all_token_offsets[doc_idx]

        answer_spans = []
        if with_answers:
            assert isinstance(answers, list)
            for a in answers:
                if a.doc_idx != doc_idx:
                    continue

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
        support_length = all_support_length[doc_idx]
        if max_support_length is not None and support_length > max_support_length > 0:
            if max_answer < max_support_length:
                # Find new start and end in the flattened support
                new_start = 0
                new_end = max_support_length
            else:
                offset = rng.randint(1, 11)
                new_end = max_answer
                new_start = max(0, min(min_answer, new_end + 2 * offset - max_support_length))
                while new_end - new_start > max_support_length - 2 * offset:
                    answer_spans = [(s, e) for s, e in answer_spans if e < new_end]
                    new_end = max(answer_spans, key=lambda span: span[1])[1]
                    new_start = max(0, min(min_answer, new_end + 2 * offset - max_support_length))
                new_end = min(new_end + offset, support_length)
                new_start = max(new_start - offset, 0)

            # Crop support according to new start and end pointers
            all_support_tokens[doc_idx] = support_tokens[new_start:new_end]
            all_support_ids[doc_idx] = all_support_ids[doc_idx][new_start:new_end]
            if with_lemmas:
                all_support_lemmas[doc_idx] = all_support_lemmas[doc_idx][new_start:new_end]
            answer_spans = [(s - new_start, e - new_start) for s, e in answer_spans]
            all_word_in_question[doc_idx] = all_word_in_question[doc_idx][new_start:new_end]
            all_support_length[doc_idx] = new_end - new_start
            all_token_offsets[doc_idx] = token_offsets[new_start:new_end]
        all_answer_spans.append(answer_spans)

    return question_tokens, question_ids, question_lemmas, question_length, \
           all_support_tokens, all_support_ids, all_support_lemmas, all_support_length, \
           all_word_in_question, all_token_offsets, all_answer_spans
