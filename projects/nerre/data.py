import os
from collections import defaultdict
from typing import NamedTuple, Sequence, Mapping

# load training, dev and test data
train_dir = "/Users/riedel/corpora/scienceie/train2"
dev_dir = "/Users/riedel/corpora/scienceie/dev/"

Token = NamedTuple("Token", [("token_start", int),
                             ("token_end", int),
                             ("word", str)])

Sentence = NamedTuple("Sentence", [("tokens", Sequence[Token])])

Keyphrase = NamedTuple("KeyPhrase", [("start", int), ("end", int), ("type", str)])

Instance = NamedTuple("Instance", [("doc", Sequence[Sentence]), ("labels", Mapping[Keyphrase, Token])])


def read_ann(textfolder=dev_dir):
    '''
    Read .ann files and look up corresponding spans in .txt files
    :param textfolder:
    :return: tokens with character offsets
    '''
    from nltk import sent_tokenize, word_tokenize
    flist = os.listdir(textfolder)
    instances = []
    for f in flist:
        if not f.endswith(".ann"):
            continue
        f_anno = open(os.path.join(textfolder, f), "rU")
        f_text = open(os.path.join(textfolder, f.replace(".ann", ".txt")), "rU")

        # there's only one line, as each .ann file is one text paragraph
        for l in f_text:
            text = l

        sents = sent_tokenize(text)
        offset_to_token = {}
        doc = []
        for s in sents:
            tokens = word_tokenize(s)
            # recover spans for each token
            current_token_index = 0
            char_index = 0
            within_token_index = 0
            token = tokens[current_token_index]
            result_tokens = []
            start_offset = 0
            # print(tokens)
            while char_index < len(text):
                while within_token_index < len(token) and text[char_index] == token[within_token_index]:
                    char_index += 1
                    within_token_index += 1
                if within_token_index == len(token):
                    rich_token = Token(start_offset, char_index, token)
                    result_tokens.append(rich_token)
                    for offset in range(start_offset, char_index):
                        offset_to_token[offset] = rich_token
                    if current_token_index < len(tokens) - 1:
                        current_token_index += 1
                        token = tokens[current_token_index]
                    else:
                        char_index = len(text)
                else:
                    char_index += 1

                within_token_index = 0
                start_offset = char_index
            assert len(tokens) == len(result_tokens)
            assert [t.word for t in result_tokens] == tokens
            doc.append(Sentence(result_tokens))

        # mapping from tokens to their labels
        token_to_labels = defaultdict(set)
        labels_to_tokens = defaultdict(set)

        for l in f_anno:
            anno_inst = l.strip("\n").split("\t")
            if len(anno_inst) == 3:
                anno_inst1 = anno_inst[1].split(" ")
                if len(anno_inst1) == 3:
                    keytype, start, end = anno_inst1
                else:
                    keytype, start, _, end = anno_inst1
                if not keytype.endswith("-of"):

                    # look up span in text and print error message if it doesn't match the .ann span text
                    keyphr_text_lookup = text[int(start):int(end)]
                    keyphr_ann = anno_inst[2]
                    if keyphr_text_lookup != keyphr_ann:
                        print("Spans don't match for anno " + l.strip() + " in file " + f)
                    for offset in range(int(start), int(end)):
                        token = offset_to_token.get(offset, None)
                        if token:
                            token_to_labels[token].add(keytype)
                            labels_to_tokens[Keyphrase(int(start), int(end), keytype)].add(token)
        instances.append(Instance(doc, labels_to_tokens))
        # print(Instance(doc, labels_to_tokens))
    return instances


instances = read_ann(dev_dir)

print(instances[0].labels)
