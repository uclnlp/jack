from quebap.sisyphos.pipelines import pipeline
from quebap.sisyphos.vocab import Vocab
from pprint import pprint


def vocab_test():
    train_data = {
        'candidates': [['entailment', 'neutral', 'contradiction']],
        'answers': ['neutral'],
        'question': ['A person is training his horse for a competition.'],
        'support': ['A person on a horse jumps over a broken down airplane.']}

    print('build vocab based on train data')
    _, train_vocab, train_answer_vocab, train_candidate_vocab = \
        pipeline(train_data, normalize=True)

    pprint(train_vocab.sym2freqs)
    pprint(train_vocab.sym2id)

    MIN_VOCAB_FREQ = 5
    train_vocab = train_vocab.prune(MIN_VOCAB_FREQ)

    pprint(train_vocab.sym2freqs)
    pprint(train_vocab.sym2id)

    print('encode train data')
    train_data, _, _, _ = pipeline(train_data, train_vocab, train_answer_vocab, train_candidate_vocab,
                                normalize=True, freeze=True)
    print(train_data)


if __name__ == "__main__":
    vocab_test()
