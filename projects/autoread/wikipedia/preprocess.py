# Preprocesses wikipedia dump cleaned and extracted by the WikiExtractor (https://github.com/attardi/wikiextractor)
from collections import defaultdict

import spacy
import os


def preprocess_files_recursively(dir, output, num_train_chunks=100, num_valid_chunks=1, newline_token="</S> <S>"):
    num_chunks = num_train_chunks + num_valid_chunks
    chunk_id = 0
    nlp = spacy.en.English(tagger=False, parser=False, entity=False, matcher=False, serializer=False, load_vectors=False)
    if not os.path.exists(output):
        os.mkdir(output)
    writers = [open(os.path.join(output, "%s_%d.txt" %
                                 ("train" if num_valid_chunks <= i else "valid", i-num_valid_chunks if num_valid_chunks <= i else i)), "w") for i in range(num_chunks)]
    context = None
    word_counts = defaultdict(lambda: 0)
    last_token = br_token.split(" ")[0]
    start_token = br_token.split(" ")[1] if " " in br_token else None
    for sub_dir, _, files in os.walk(dir):
        for fn in files:
            print("Processing %s" % fn)
            fn = os.path.join(sub_dir, fn)
            with open(fn, 'rb') as f:
                for l in f:
                    l = l.decode("utf-8")
                    if l.startswith("</doc>"): #new document
                        while context and context[-1] == br_token:
                            context = context[:-1]
                        context.append(last_token)
                        context = ' '.join(context)+"\n"
                        writers[chunk_id].write(context)
                        for w in context.split(' '):
                            word_counts[w] += 1
                        chunk_id = (chunk_id + 1) % num_chunks
                    elif l.startswith("<"):
                        if start_token is None:
                            context = []
                        else:
                            context = [start_token]
                    else:
                        processed = nlp(l.strip())
                        for token in processed:
                            if not token.is_space:
                                context.append(token.text)
                        context.append(newline_token)

    for w in writers:
        w.close()

   # word_counts = sorted(list(word_counts.items()), key=lambda x: -x[1])
    #with open(os.path.join(output, "document.vocab"), "w") as f:
    #    for i, (token, count) in enumerate(word_counts):
    #        f.write("%d\t%s\t%d" % (i, token, count))
    #        if i < len(word_counts)-1:
    #            f.write("\n")


if __name__ == '__main__':
    import sys
    dir = sys.argv[1]
    out = sys.argv[2]
    num_train_chunks = int(sys.argv[3])
    num_valid_chunks = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    br_token = sys.argv[5] if len(sys.argv) > 5 else "</S> <S>"

    preprocess_files_recursively(dir, out, num_train_chunks, num_valid_chunks, br_token)




