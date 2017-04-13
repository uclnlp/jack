import random
import re

from jtr.jack.data_structures import QASetting
from jtr.preprocess.map import deep_map

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


def prepare_data(dataset, vocab, lowercase=False, with_answers=False, wiq_contentword=False,
                 with_spacy=False, max_support_length=-1):
    if with_spacy:
        import spacy
        nlp = spacy.load("en", parser=False)
        thistokenize = lambda t: nlp(t)
    else:
        thistokenize = tokenize

    corpus = {"support": [], "question": []}
    for d in dataset:
        if isinstance(d, QASetting):
            qa_setting = d
        else:
            qa_setting, answer = d

        if lowercase:
            corpus["support"].append(" ".join(qa_setting.support).lower())
            corpus["question"].append(qa_setting.question.lower())
        else:
            corpus["support"].append(" ".join(qa_setting.support))
            corpus["question"].append(qa_setting.question)

    corpus_tokenized = deep_map(corpus, thistokenize, ['question', 'support'])

    word_in_question = []
    question_lengths = []
    support_lengths = []
    token_offsets = []
    answer_spans = []

    rng = random.Random(12345)

    for i, (q, s) in enumerate(zip(corpus_tokenized["question"], corpus_tokenized["support"])):
        # word in question feature
        wiq = []
        for token in s:
            if with_spacy:
                wiq.append(float(any(token.lemma == t2.lemma for t2 in q) and
                                 (not wiq_contentword or (token.orth_.isalnum() and not token.is_stop))))
            else:
                wiq.append(float(token in q and (not wiq_contentword or token.isalnum())))
        word_in_question.append(wiq)

        if with_spacy:
            offsets = [t.idx for t in s]
            s = [t.orth_ for t in s]
            q = [t.orth_ for t in q]
            corpus_tokenized["question"][i] = q
            corpus_tokenized["support"][i] = s
        else:
            # char to token offsets
            support = corpus["support"][i]
            offsets = token_to_char_offsets(support, s)

        token_offsets.append(offsets)
        question_lengths.append(len(q))

        min_answer = len(s)
        max_answer = 0

        spans = []
        if with_answers:
            answers = dataset[i][1]
            for a in answers:
                start = 0
                while start < len(offsets) and offsets[start] < a.span[0]:
                    start += 1

                if start == len(offsets):
                    continue

                end = start
                while end + 1 < len(offsets) and offsets[end + 1] < a.span[1]:
                    end += 1
                if (start, end) not in spans:
                    spans.append((start, end))
                    min_answer = min(min_answer, start)
                    max_answer = max(max_answer, end)

        # cut support whenever there is a maximum allowed length and recompute answer spans
        if max_support_length is not None and max_support_length > 0 and len(s) > max_support_length:
            if max_answer < max_support_length:
                s = s[:max_support_length]
                word_in_question[-1] = word_in_question[-1][:max_support_length]
            else:
                offset = rng.randint(1, 11)
                new_end = max_answer + offset
                new_start = max(0, min(min_answer - offset, new_end - max_support_length))
                while new_end - new_start > max_support_length:
                    spans = [(s, e) for s, e in spans if e < (new_end - offset)]
                    new_end = max(spans, key=lambda span: span[1])[1] + offset
                    new_start = max(0, min(min_answer - offset, new_end - max_support_length))
                s = s[new_start:new_end]
                spans = [(s - new_start, e - new_start) for s, e in spans]
                word_in_question[-1] = word_in_question[-1][new_start:new_end]

            corpus_tokenized["support"][i] = s
            answer_spans.append(spans)

        else:
            answer_spans.append(spans)

        support_lengths.append(len(s))

    corpus_ids = deep_map(corpus_tokenized, vocab, ['question', 'support'])

    return corpus_tokenized["question"], corpus_ids["question"], question_lengths, \
           corpus_tokenized["support"], corpus_ids["support"], support_lengths, \
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
