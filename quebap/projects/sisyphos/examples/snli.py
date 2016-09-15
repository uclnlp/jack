import json
from pprint import pprint
from sisyphos.io import map_corpus, tokenize, lower, corpus_to_ids, \
    seqs_to_words, seqs_to_ids, Vocab, seq_to_ids

target_vocab = Vocab({"contradiction": 1, "neutral": 2, "entailment": 3},
                     ["contradiction", "neutral", "entailment"])


def load(path):
    seq1s = []
    seq2s = []
    targets = []
    with open(path, "r") as f:
        for line in f.readlines():
            instance = json.loads(line.strip())
            sentence1 = instance['sentence1']
            sentence2 = instance['sentence2']
            target = instance['gold_label']
            if target != "-":
                seq1s.append(sentence1)
                seq2s.append(sentence2)
                targets.append(target)
    return [seq1s, seq2s, targets]


def pipeline(corpus, vocab=None, freeze=False):
    # not tokenizing labels
    corpus_tokenized = map_corpus(corpus, tokenize, [0, 1])
    corpus_lower = map_corpus(corpus_tokenized, lower, [0, 1])
    corpus_ids, vocab = corpus_to_ids(corpus_lower, [0, 1], vocab, freeze)

    target_ids, _ = seq_to_ids(corpus_ids[2], target_vocab, freeze)
    corpus_ids[2] = target_ids
    return corpus_ids, vocab


if __name__ == '__main__':
    train, dev, test = [
        load("./data/snli/snli_1.0/snli_1.0_%s.jsonl" % name)
        for name in ["train", "dev", "test"]]

    train, train_vocab = pipeline(train)
    dev, _ = pipeline(dev, train_vocab, freeze=True)
    test, _ = pipeline(test, train_vocab, freeze=True)

    print(train[0][0])
    print(train[2][0])

    for i in range(10):
        print(seqs_to_words([dev[0][i], dev[1][i]], train_vocab), dev[2][i])

