from collections import namedtuple, defaultdict
import re
from pprint import pprint

Vocab = namedtuple("Vocab", ["word2id", "id2word"])

# sym (e.g. token, token id or class label)
# seq (e.g. sequence of tokens)
# seqs (sequence of sequences)
# corpus (sequence of sequence of sequences)
#   e.g. hypotheses (sequence of sequences)
#        premises (sequence of sequences)
#        labels (sequence of symbols)
# corpus = [hypotheses, premises, labels]


def map_seqs(seqs, fun):
    return [fun(seq) for seq in seqs]


def map_corpus(corpus, fun, indices=None):
    corpus_mapped = []
    for i, seqs in enumerate(corpus):
        if indices is None or i in indices:
            corpus_mapped.append(map_seqs(seqs, fun))
        else:
            corpus_mapped.append(seqs)
    return corpus_mapped


def tokenize(seq, pattern="([\s'\-\.\!])"):
    return [x for x in re.split(pattern, seq)
            if not re.match("\s", x) and x != ""]


def lower(seq):
    return [x.lower() for x in seq]


def seq_to_ids(seq, vocab=None, freeze=False):
    if vocab is None:
        id2word = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<GO>"]
        word2id = defaultdict(lambda: 1)  # defaults to <UNK>
        for i, word in enumerate(id2word):
            word2id[word] = i
        vocab = Vocab(word2id, id2word)

    seq_ids = []
    for word in seq:
        if word in vocab.word2id:
            seq_ids.append(vocab.word2id[word])
        else:
            if not freeze:
                vocab.word2id[word] = len(vocab.id2word)
                vocab.id2word.append(word)
                seq_ids.append(vocab.word2id[word])
            else:
                seq_ids.append(vocab.word2id["<UNK>"])
    return seq_ids, vocab


def seqs_to_ids(seqs, vocab=None, freeze=False):
    seqs_ids = []
    for seq in seqs:
        seq_ids, vocab = seq_to_ids(seq, vocab)
        seqs_ids.append(seq_ids)
    return seqs_ids, vocab


def corpus_to_ids(corpus, indices=None, vocab=None, freeze=False):
    corpus_ids = []
    for i, seqs in enumerate(corpus):
        if indices is None or i in indices:
            seqs_ids, vocab = seqs_to_ids(seqs, vocab, freeze)
            corpus_ids.append(seqs_ids)
        else:
            corpus_ids.append(seqs)
    return corpus_ids, vocab


def seqs_to_words(seqs, vocab):
    def inner(seq):
        return [vocab.id2word[x] for x in seq]
    return map_seqs(seqs, inner)


if __name__ == '__main__':
    data = [
        [
            "All work and no play makes Jack a dull boy.",
            "All work and no play makes Jack a dull boy",
            "All work and no-play makes Jack a dull boy"
        ],
        [
            "I'm sorry Dave, I'm afraid I can't do that!",
            "I'm sorry Dave, I'm afraid I can't do that",
            "I'm sorry Dave, I'm afraid I can't do that"
        ]
    ]

    data_tokenized = map_corpus(data, tokenize)
    data_lower = map_corpus(data_tokenized, lower)
    data_ids, vocab = corpus_to_ids(data_lower)

    print(data)
    print(data_tokenized)
    print(data_lower)
    print(data_ids)
    print(vocab)
    print([seqs_to_words(seqs, vocab) for seqs in data_ids])

    print(vocab.word2id["afraid"])
    print(vocab.word2id["hal-9000"])  # <UNK>
