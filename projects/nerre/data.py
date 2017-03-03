import os
from collections import defaultdict
from typing import NamedTuple, Sequence, Mapping, Tuple
from random import randint

# load training, dev and test data
from jtr.preprocess.batch import get_batches
from jtr.preprocess.map import deep_map
from jtr.preprocess.vocab import Vocab
import numpy as np
from projects.nerre.eval import calculateMeasures

train_dir = "/Users/Isabelle/Documents/UCLMR/semeval2017-orga/data/train2"
dev_dir = "/Users/Isabelle/Documents/UCLMR/semeval2017-orga/data/dev/"
test_dir = "/Users/Isabelle/Documents/UCLMR/semeval_articles/test_final2/"

Token = NamedTuple("Token", [("token_start", int),
                             ("token_end", int),
                             ("word", str),
                             ("index", int)])

Sentence = NamedTuple("Sentence", [("tokens", Sequence[Token])])

Keyphrase = NamedTuple("KeyPhrase", [("start", int), ("end", int), ("type", str), ("text", str), ("id", str)])

Instance = NamedTuple("Instance", [("text", str),
                                   ("doc", Sequence[Sentence]),
                                   ("labels", Mapping[Keyphrase, Sequence[Token]]),
                                   ("file_name", str),
                                   ("relations", Sequence[Tuple[str, Keyphrase, Keyphrase]])])

bio_vocab = Vocab(unk="O")
bio_vocab("B")
bio_vocab("I")
bio_vocab("O")
bio_vocab.freeze()

label_vocab = Vocab(unk="O")
label_vocab("Material")
label_vocab("Process")
label_vocab("Task")
label_vocab.freeze()

rel_type_vocab = Vocab(unk="O")
rel_type_vocab("Hyponym-of")
rel_type_vocab("Synonym-of")
rel_type_vocab("O")
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
    #i = 0
    for f in flist:
        if not f.endswith(".ann"):
            continue
        #i += 1
        #if i == 3:
        #    break
        #if not f == "S0003491613001516.ann": #S0021999113005846.ann":  # good test example
        #    continue
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
        keyphrase_to_tokens = defaultdict(set)
        relations = []
        id_to_keyphrase = {}

        resolved_relations = []

        for l in f_anno:
            anno_inst = l.strip("\n").split("\t")

            if len(anno_inst) == 3 or len(anno_inst) == 2:
                anno_inst1 = anno_inst[1].split(" ")
                annotation_id = anno_inst[0].strip()
                if len(anno_inst1) == 3:
                    keytype, start, end = anno_inst1
                elif len(anno_inst1) > 3 and anno_inst1[0] == "Synonym-of":
                    for i in range(1, len(anno_inst1)-1):
                        relations.append((annotation_id, (keytype, anno_inst1[1], anno_inst1[1+i])))
                    keytype = anno_inst1[0]
                else:
                    keytype = anno_inst1[0]
                    start = anno_inst1[1]
                    end = anno_inst1[len(anno_inst1)-1]
                if not keytype.endswith("-of"):

                    # look up span in text and print error message if it doesn't match the .ann span text
                    keyphr_text_lookup = text[int(start):int(end)]
                    keyphr_ann = anno_inst[2]
                    keyphrase = Keyphrase(int(start), int(end), keytype, keyphr_text_lookup, annotation_id)
                    id_to_keyphrase[annotation_id] = keyphrase
                    try:
                        assert keyphr_text_lookup == keyphr_ann
                        for offset in range(int(start), int(end)):
                            token = offset_to_token.get(offset, None)
                            if token:
                                token_to_labels[token].add(keytype)
                                keyphrase_to_tokens[keyphrase].add(token)
                    except AssertionError:
                        print("Span lookup doesn't match tokens:", keyphr_text_lookup, "    vs   " , keyphr_ann)
                elif keytype.endswith("-of") and len(anno_inst1) == 3:  # otherwise we already took care of that
                    #anno_inst1 = (keytype, start, end)
                    relations.append((annotation_id, (keytype, start, end))) #anno_inst1))
        for annotation_id, (rel, arg1, arg2) in relations:
            if rel == "Hyponym-of":
                arg1_id = arg1.split(":")[1]
                arg2_id = arg2.split(":")[1]
            else:
                arg1_id = arg1
                arg2_id = arg2
            arg1_kp = id_to_keyphrase[arg1_id]
            arg2_kp = id_to_keyphrase[arg2_id]
            resolved_relations.append((rel, arg1_kp, arg2_kp))

        sorted_kp_to_tokens = {}
        for kp, tokens in keyphrase_to_tokens.items():
            sorted_kp_to_tokens[kp] = sorted(tokens, key=lambda t: t.token_start)

        instances.append(Instance(text, doc, sorted_kp_to_tokens, f, resolved_relations))
        # print(Instance(doc, keyphrase_to_tokens))
    print("Collected {} word types".format(len(word_types)))
    return instances


def convert_to_batchable_format(instances, vocab,
                                sentences_as_ints_ph="sentences_as_ints",
                                document_indices_ph="document_indices",
                                bio_labels_as_ints_ph="bio_labels_as_ints",
                                type_labels_as_ints_ph="type_labels_as_ints",
                                token_char_offsets_ph="token_char_offsets",
                                relation_matrices_ph="relation_matrices",
                                sentence_lengths_ph="sentence_length"):
    # convert
    sentences_as_ints = []
    bio_labels_as_ints = []
    type_labels_as_ints = []
    document_indices = []
    token_char_offsets = []
    relation_matrices = []
    sentence_lengths = []
    for doc_index, instance in enumerate(instances):
        for sentence in instance.doc:
            sentences_as_ints.append([vocab(token.word) for token in sentence.tokens])
            token_char_offsets.append([[token.token_start, token.token_end] for token in sentence.tokens])
            document_indices.append(doc_index)
            sentence_lengths.append(len(sentence.tokens))
            bio_labels = [bio_vocab("O")] * len(sentence.tokens)
            type_labels = [label_vocab("O")] * len(sentence.tokens)
            sentence_tokens = set(sentence.tokens)

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

            relation_matrix = np.ndarray([len(sentence.tokens), len(sentence.tokens)], dtype=np.int32)
            relation_matrix.fill(rel_type_vocab("O"))

            for rel, arg1, arg2 in instance.relations:
                try:
                    if instance.labels[arg1][0] in sentence_tokens and instance.labels[arg2][0] in sentence_tokens:
                        # for first token of ent1 and ent2, add corresponding rel type to matrix
                        tok1 = instance.labels[arg1][0]
                        tok2 = instance.labels[arg2][0]
                        relation_matrix[tok1.index, tok2.index] = rel_type_vocab(rel)
                except KeyError:
                    print("No corresponding tokens found:", rel, arg1, arg2)
                #else:
                #    print("No corresponding tokens found:", rel, arg1, arg2)  # that's ok - they might be found in other sentences
            relation_matrices.append(relation_matrix.tolist())

    return {
        sentences_as_ints_ph: sentences_as_ints,
        document_indices_ph: document_indices,
        bio_labels_as_ints_ph: bio_labels_as_ints,
        type_labels_as_ints_ph: type_labels_as_ints,
        token_char_offsets_ph: token_char_offsets,
        relation_matrices_ph: relation_matrices,
        sentence_lengths_ph: sentence_lengths

    }


def fill_vocab(instances, vocab):
    for instance in instances:
        for sent in instance.doc:
            for token in sent.tokens:
                vocab(token.word)


def convert_batch_to_ann(batch, instances, out_dir="/tmp",
                         sentences_as_ints_key="sentences_as_ints",
                         document_indices_key="document_indices",
                         bio_labels_key="bio_labels_as_ints",
                         type_labels_as_ints_key="type_labels_as_ints",
                         token_char_offsets_key="token_char_offsets",
                         relation_matrices_key="relation_matrices",
                         sentence_lengths_key="sentence_length"):
    doc_id_to_doc_info = {}
    doc_ids = batch[document_indices_key]
    bio_labels = batch[bio_labels_key]
    type_labels = batch[type_labels_as_ints_key]
    relation_matrices = batch[relation_matrices_key]
    token_char_offsets = batch[token_char_offsets_key]
    sentence_lengths = batch[sentence_lengths_key]

    prev_filename = ""
    for elem_index, doc_id in enumerate(doc_ids):
        instance = instances[doc_id]
        if instance.file_name != prev_filename:
            doc_info = doc_id_to_doc_info.get(instance.file_name, {"kps": {}, "rels": []})
            current_kps = doc_info["kps"]
            current_relations = doc_info["rels"]
        in_kp = False
        last_symbol = "O"
        kp_type = None
        sentence_length = sentence_lengths[elem_index]
        #print(instance.file_name)
        #print(instance)
        kps_in_sentence = []
        relation_matrix = relation_matrices[elem_index]
        bio_label_sequence = bio_labels[elem_index]
        type_label_sequence = type_labels[elem_index]
        token_char_offset_sequence = token_char_offsets[elem_index]
        current_kp_start = token_char_offset_sequence[0][0]  # previously: O
        current_kp_end = -1

        char_offset_to_token_index = {}

        def create_kp():
            text = instance.text[current_kp_start:current_kp_end]
            kp_id = "T" + str(len(current_kps) + 1)
            kp = Keyphrase(current_kp_start, current_kp_end, kp_type, text, kp_id)
            current_kps[kp_id] = kp
            kps_in_sentence.append(kp)


        for token_index, (bio_label, type_label, (start, end)) in enumerate(
                zip(bio_label_sequence[:sentence_length],
                    type_label_sequence[:sentence_length],
                    token_char_offset_sequence[:sentence_length])):
            bio_label_symbol = bio_vocab.get_sym(bio_label)
            type_label_symbol = label_vocab.get_sym(type_label)
            for i in range(start, end):
                char_offset_to_token_index[i] = token_index

            # !!! introduced this first conditioned new here, solves some problems
            if bio_label_symbol == "B" and (last_symbol == "B" or last_symbol == "I") and type_label_symbol != "O":
                create_kp()
                current_kp_start = start
                kp_type = type_label_symbol
                in_kp = True
            elif bio_label_symbol == "B" and type_label_symbol != "O":
                if in_kp:
                    create_kp()
                else:
                    in_kp = True
                    current_kp_start = start
                    kp_type = type_label_symbol
            elif bio_label_symbol == "O":
                if last_symbol != bio_label_symbol:
                    create_kp()
                    in_kp = False

            current_kp_end = end
            last_symbol = bio_label_symbol
        if last_symbol != "O":
            create_kp()

        #print(min(char_offset_to_token_index, key=char_offset_to_token_index.get), max(char_offset_to_token_index, key=char_offset_to_token_index.get))
        #if len(current_kps) > 0:
            #print(current_kps)
        # now find relations
        for kp1 in kps_in_sentence:
            for kp2 in kps_in_sentence:
                if kp1 != kp2:
                    # find the first token of both key phrases
                    #try:
                        tok1 = char_offset_to_token_index[kp1.start]
                        tok2 = char_offset_to_token_index[kp2.start]
                        relation = rel_type_vocab.get_sym(relation_matrix[tok1, tok2])
                        if relation != "O" and tok1 != tok2:
                            current_relations.append((relation, kp1.id, kp2.id))
                    #except KeyError:
                    #    print("Key Error!", kp1.start, kp1.end, kp2.start, kp2.end, max(char_offset_to_token_index, key=char_offset_to_token_index.get))
                    #    continue

        # !!! this was previously overwritten for every sentence
        doc_id_to_doc_info[instance.file_name] = doc_info

        #if len(current_relations) > 0:
            #print(current_relations)

        prev_filename = instance.file_name

    for file_name, doc_info in doc_id_to_doc_info.items():
        with open(out_dir + "/" + file_name, "a") as ann:  # !!! this was previously "w" -> not all sentences for each file are in the batch together, got overwritten
            for key, kp in doc_info["kps"].items():
                ann.write("{key}\t{label} {start} {end}\t{text}\n".format(key=key,
                                                                          label=kp.type,
                                                                          start=kp.start,
                                                                          end=kp.end,
                                                                          text=kp.text))
            for rel, arg1, arg2 in doc_info["rels"]:
                ann.write("*\t{label} {arg1} {arg2}\n".format(label=rel, arg1=arg1, arg2=arg2))  # !!! \n was missing

def reset_output_dir():
    # reset current out_dir
    out_dir = "/tmp/"
    for f in os.listdir(out_dir):
        if os.path.isfile(os.path.join(out_dir, f)) and f.endswith(".ann"):
            os.remove(os.path.join(out_dir, f))

def randomBaseline(batches):

        #dev_pred_batches_i["bio_labels_as_ints"], dev_pred_batches_i["type_labels_as_ints"], \
        #dev_pred_batches_i["relation_matrices"]
    for i, batch in enumerate(batches["bio_labels_as_ints"]):
        for ii, tok in enumerate(batch):
            batches["bio_labels_as_ints"][i][ii] = randint(0, len(bio_vocab)-1)
    for j, batch in enumerate(batches["type_labels_as_ints"]):
        for jj, tok in enumerate(batch):
            batches["type_labels_as_ints"][j][jj] = randint(0, len(label_vocab) - 1)
    for batch in batches["relation_matrices"]:
        for k, seq in enumerate(batch):
            if type(k) != list:
                continue
            for kk in seq:
                batches["relation_matrices"][k][kk] = randint(0, len(rel_type_vocab) - 1)
    return batches


if __name__ == "__main__":

    reset_output_dir()

    vocab = Vocab()
    instances = read_ann(test_dir)
    fill_vocab(instances, vocab)
    batchable = convert_to_batchable_format(instances, vocab)  #[:2]

    # random baseline
    batchable = randomBaseline(batchable)

    #print(batchable)
    batches = list(get_batches(batchable))#[:2]
    for batch in batches:
        convert_batch_to_ann(batch, instances, "/tmp")

    calculateMeasures(test_dir, "/tmp/")

# print(instances[0].labels)
