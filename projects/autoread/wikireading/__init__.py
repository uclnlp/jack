import tensorflow as tf

tfrecord_features = {
    "answer_breaks": tf.VarLenFeature(tf.int64),
    "answer_ids": tf.VarLenFeature(tf.int64),
    "answer_location": tf.VarLenFeature(tf.int64),
    "answer_sequence": tf.VarLenFeature(tf.int64),
    "answer_string_sequence": tf.VarLenFeature(tf.string),
    "break_levels": tf.VarLenFeature(tf.int64),
    "document_sequence": tf.VarLenFeature(tf.int64),
    "full_match_answer_location": tf.VarLenFeature(tf.int64),
    "paragraph_breaks": tf.VarLenFeature(tf.int64),
    "question_sequence": tf.VarLenFeature(tf.int64),
    "question_string_sequence": tf.VarLenFeature(tf.string),
    "raw_answer_ids": tf.VarLenFeature(tf.int64),
    "raw_answers": tf.VarLenFeature(tf.string),
    "sentence_breaks": tf.VarLenFeature(tf.int64),
    "string_sequence": tf.VarLenFeature(tf.string),
    "type_sequence": tf.VarLenFeature(tf.int64)
}


""" Largest answer
A History of the Clan MacLean from Its First Settlement at Duard Castle, in the Isle of Mull, to the Present Period: Including a Genealogical Account of Some of the Principal Families Together with Their Heraldry, Legends, Superstitions, etc.
"""
max_answer_length = 46 + 2 # including <S> and </S>
max_question_length = 10

def load_vocab(fn):
    ids = dict()
    counts = dict()
    vocab = list()
    with open(fn, "rb") as f:
        for l in f:
            [i, word, count] = l.decode("utf-8").strip().split("\t")
            counts[word] = count
            i = int(i)
            if i == len(vocab):
                vocab.append(word)
            else:
                vocab[i] = word
            ids[word] = i
    return ids, vocab, counts