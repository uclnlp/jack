import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

from jack.util.preprocessing import sort_by_tfidf


def extract_support(triviaqa_question, docs, corpus, max_num_support, max_tokens):
    answers = []
    supports = []
    paragraph_tokens = []
    separator = '$|$'
    for doc in docs:
        doc_tokens = corpus.get_document(doc.doc_id)
        doc_tokens_flat = [t for p in doc_tokens for s in p for t in s]
        doc_paragraph_tokens = [[t for s in p for t in s] for p in doc_tokens]

        # merge many small paragraphs
        if max_tokens > 0:
            new_paragraph_tokens = [[]]
            for s in doc_paragraph_tokens:
                if len(new_paragraph_tokens[-1]) + len(s) >= max_tokens and len(new_paragraph_tokens[-1]) > 0:
                    # start new paragraph
                    if len(s) >= max_tokens:
                        while s:
                            new_paragraph_tokens.append(s[:max_tokens])
                            s = s[max_tokens:]
                    else:
                        new_paragraph_tokens.append(s)
                else:
                    # merge with recent paragraph
                    if len(new_paragraph_tokens[-1]) > 0:
                        new_paragraph_tokens[-1].append(separator)
                    new_paragraph_tokens[-1].extend(s)
        else:
            new_paragraph_tokens = doc_paragraph_tokens
        paragraph_tokens.extend(new_paragraph_tokens)

        p_idx_flat = [i for i, p in enumerate(new_paragraph_tokens) for t in p if t != separator]
        assert len(doc_tokens_flat) == len(p_idx_flat)

        doc_idx_offset = len(supports)
        supports.extend(" ".join(s) for s in new_paragraph_tokens)
        support_offsets = [0]
        for s in supports[doc_idx_offset:]:
            support_offsets.append(support_offsets[-1] + len(s) + 1)
        if doc.answer_spans is not None:
            for flat_s, flat_e in doc.answer_spans:
                p_idx = p_idx_flat[flat_s]
                s = flat_s - sum(1 for p2 in new_paragraph_tokens[:p_idx] for t in p2 if t != separator)
                p = new_paragraph_tokens[p_idx]
                k = 0
                char_s = 0
                while s > k:
                    if p[k] == separator:
                        s += 1
                    char_s += len(p[k]) + 1
                    k += 1
                char_e = char_s + sum(len(t) + 1 for t in doc_tokens_flat[flat_s:flat_e + 1]) - 1
                answers.append({
                    "text": " ".join(doc_tokens_flat[flat_s:flat_e + 1]),
                    "span": [char_s, char_e],
                    "doc_idx": p_idx + doc_idx_offset
                })
        del doc_tokens, p_idx_flat, doc_tokens_flat

    if max_num_support > 0 and len(supports) > max_num_support:
        sorted_supports = sort_by_tfidf(" ".join(triviaqa_question.question), [' '.join(p) for p in paragraph_tokens])
        sorted_supports = [i for i, _ in sorted_supports]
        sorted_supports_rev = {v: k for k, v in enumerate(sorted_supports)}
        if answers:
            min_answer_rev = min(sorted_supports_rev[a['doc_idx']] for a in answers)
            if min_answer_rev >= max_num_support:
                min_answer = sorted_supports[min_answer_rev]
                # force at least one answer by swapping best paragraph with answer to be the n-th paragraph
                old_nth_best = sorted_supports[max_num_support - 1]
                sorted_supports[min_answer_rev] = sorted_supports[max_num_support - 1]
                sorted_supports[max_num_support - 1] = min_answer
                sorted_supports_rev[old_nth_best] = min_answer_rev
                sorted_supports_rev[min_answer] = max_num_support - 1

        sorted_supports_rev = {v: k for k, v in enumerate(sorted_supports)}
        supports = [supports[i] for i in sorted_supports[:max_num_support]]
        is_an_answer = len(answers) > 0
        answers = [a for a in answers if sorted_supports_rev[a['doc_idx']] < max_num_support]
        for a in answers:
            a['doc_idx'] = sorted_supports_rev[a['doc_idx']]
        assert not is_an_answer or len(answers) > 0

    return supports, answers


def convert_triviaqa(triviaqa_question, corpus, max_num_support, max_tokens, is_web):
    question = " ".join(triviaqa_question.question)
    if is_web:
        for doc in triviaqa_question.web_docs:
            supports, answers = extract_support(triviaqa_question, [doc], corpus, max_num_support, max_tokens)
            filename = corpus.file_id_map[doc.doc_id]
            question_id = triviaqa_question.question_id + '--' + filename[4:] + ".txt"
            yield {"questions": [{"answers": answers,
                                  "question": {"text": question, "id": question_id}}],
                   "support": supports}
        for doc in triviaqa_question.entity_docs:
            supports, answers = extract_support(triviaqa_question, [doc], corpus, max_num_support, max_tokens)
            question_id = triviaqa_question.question_id + '--' + doc.title.replace(' ', '_') + ".txt"
            yield {"questions": [{"answers": answers,
                                  "question": {"text": question, "id": question_id}}],
                   "support": supports}
    else:
        question_id = triviaqa_question.question_id
        supports, answers = extract_support(triviaqa_question, triviaqa_question.entity_docs,
                                            corpus, max_num_support, max_tokens)
        yield {"questions": [{"answers": answers,
                              "question": {"text": question, "id": question_id}}],
               "support": supports}


def process(x, verbose=False):
    dataset, filemap, max_num_support, max_tokens, is_web = x
    instances = []
    corpus = TriviaQaEvidenceCorpusTxt(filemap)
    for i, q in enumerate(dataset):
        if verbose and i % 1000 == 0:
            print("%d/%d done" % (i, len(dataset)))
        instances.extend(x for x in convert_triviaqa(q, corpus, max_num_support, max_tokens, is_web))
    return instances


def convert_dataset(path, filemap, name, num_processes, max_num_support, max_tokens, is_web=True):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)

    if num_processes == 1:
        instances = process((dataset, filemap, max_num_support, max_tokens, is_web), True)
    else:
        chunk_size = 1000
        executor = ProcessPoolExecutor(num_processes)
        instances = []
        i = 0
        for processed in executor.map(
                process, [(dataset[i * chunk_size:(i + 1) * chunk_size], filemap, max_num_support, max_tokens, is_web)
                          for i in range(len(dataset) // chunk_size + 1)]):
            instances.extend(processed)
            i += chunk_size
            print("%d/%d done" % (min(len(dataset), i), len(dataset)))

    return {"meta": {"source": name}, 'instances': instances}


if __name__ == '__main__':
    from docqa.triviaqa.evidence_corpus import TriviaQaEvidenceCorpusTxt
    import json

    dataset = sys.argv[1]

    if len(sys.argv) > 2:
        num_processes = int(sys.argv[2])
    else:
        num_processes = 1

    if len(sys.argv) > 3:
        max_paragraphs = int(sys.argv[3])
    else:
        max_paragraphs = -1

    if len(sys.argv) > 4:
        max_tokens = int(sys.argv[4])
    else:
        max_tokens = -1

    triviaqa_prepro = os.environ['TRIVIAQA_HOME'] + '/preprocessed'

    is_web = dataset.startswith('web')
    dataset, split = dataset.split('-')

    ds = os.path.join(triviaqa_prepro, 'triviaqa/', dataset)
    with open(ds + "/file_map.json") as f:
        filemap = json.load(f)

    fn = '%s-%s.json' % (dataset, split)
    print("Converting %s..." % fn)
    new_ds = convert_dataset(os.path.join(ds, split + '.pkl'), filemap, fn, num_processes,
                             max_paragraphs, max_tokens, is_web)
    with open('data/triviaqa/%s' % fn, 'w') as f:
        json.dump(new_ds, f)
