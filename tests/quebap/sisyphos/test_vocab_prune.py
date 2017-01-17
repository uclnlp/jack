from quebap.sisyphos.pipelines import pipeline
from quebap.sisyphos.vocab import Vocab

def vocab_test():

    train_data = {'candidates': [['entailment', 'neutral', 'contradiction']], 'answers': ['neutral'], 'question': ['A person is training his horse for a competition.'], 'support': ['A person on a horse jumps over a broken down airplane.']}

    print('build vocab based on train data')
    train_data, train_vocab, train_answer_vocab, train_candidate_vocab = pipeline(train_data, normalize=True)

    # not working as quebap.sisyphos.vocab.Vocab.prune() function does not seem compatible with deep_map()
    MIN_VOCAB_FREQ = 2
    train_vocab = train_vocab.prune(MIN_VOCAB_FREQ)

    print('encode train data')
    train_data, _, _ = pipeline(train_data, train_vocab, train_answer_vocab, normalize=True, freeze=True)



if __name__ == "__main__":
    vocab_test()
