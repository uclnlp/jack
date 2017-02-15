import os
from collections import defaultdict
from typing import NamedTuple, Sequence, Mapping

# load training, dev and test data
from jtr.preprocess.batch import get_batches
from jtr.preprocess.map import deep_map
from jtr.preprocess.vocab import Vocab

train_dir = "/Users/riedel/corpora/scienceie/train2"
dev_dir = "/Users/riedel/corpora/scienceie/dev/"

Token = NamedTuple("Token", [("token_start", int),
                             ("token_end", int),
                             ("word", str),
                             ("index", int)])

Sentence = NamedTuple("Sentence", [("tokens", Sequence[Token])])

Keyphrase = NamedTuple("KeyPhrase", [("start", int), ("end", int), ("type", str)])

Instance = NamedTuple("Instance", [("doc", Sequence[Sentence]), ("labels", Mapping[Keyphrase, Token])])

bio_vocab = Vocab()
bio_vocab("B")
bio_vocab("I")
bio_vocab("O")
bio_vocab.freeze()

label_vocab = Vocab()
label_vocab("Material")
label_vocab("Process")
label_vocab("Task")
label_vocab.freeze()

rel_type_vocab = Vocab()
rel_type_vocab("HYPONYM_OF")
rel_type_vocab("SYNONYM_OF")
rel_type_vocab.freeze()


def read_ann(textfolder=dev_dir):
    '''
    Read .ann files and look up corresponding spans in .txt files
    :param textfolder:
    :return: tokens with character offsets
    '''
    word_types = set()
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
            word_types.update(tokens)
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
                    rich_token = Token(start_offset, char_index, token, len(result_tokens))
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
    print("Collected {} word types".format(len(word_types)))
    return instances


def convert_to_batchable_format(instances, vocab,
                                sentences_as_ints_ph="sentences_as_ints",
                                document_indices_ph="document_indices",
                                bio_labels_as_ints_ph="bio_labels_as_ints",
                                type_labels_as_ints_ph="type_labels_as_ints"):
    # convert
    sentences_as_ints = []
    bio_labels_as_ints = []
    type_labels_as_ints = []
    document_indices = []
    for doc_index, instance in enumerate(instances):
        for sentence in instance.doc:
            sentences_as_ints.append([vocab(token.word) for token in sentence.tokens])
            document_indices.append(doc_index)
            bio_labels = [bio_vocab("O")] * len(sentence.tokens)
            type_labels = [label_vocab("O")] * len(sentence.tokens)

            for kp, kp_tokens in instance.labels.items():
                started = False
                for token in sentence.tokens:
                    if token in kp_tokens:
                        type_labels[token.index] = label_vocab(kp.type)
                        if not started:
                            bio_labels[token.index] = bio_vocab("B")
                            started = True
                        else:
                            bio_labels[token.index] = bio_vocab("I")
            bio_labels_as_ints.append(bio_labels)
            type_labels_as_ints.append(type_labels)

    return {
        sentences_as_ints_ph: sentences_as_ints,
        document_indices_ph: document_indices,
        bio_labels_as_ints_ph: bio_labels_as_ints,
        type_labels_as_ints_ph: type_labels_as_ints
    }


def fill_vocab(instances, vocab):
    for instance in instances:
        for sent in instance.doc:
            for token in sent.tokens:
                vocab(token.word)


if __name__ == "__main__":
    vocab = Vocab()
    instances = read_ann(dev_dir)
    fill_vocab(instances, vocab)
    batchable = convert_to_batchable_format(instances[:1], vocab)
    print(batchable)
    batches = get_batches(batchable)
    for batch in batches:
        print(batch)

# print(instances[0].labels)
