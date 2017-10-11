# -*- coding: utf-8 -*-

from pprint import pprint

from jack.core import QASetting
from jack.util import preprocessing


def test_vocab():
    train_data = [
        QASetting(question='A person is training his horse for a competition.',
                  support=['A person on a horse jumps over a broken down airplane.'],
                  atomic_candidates=['entailment', 'neutral', 'contradiction'])
    ]

    print('build vocab based on train data')
    train_vocab = preprocessing.fill_vocab(train_data, )
    train_vocab.freeze()
    pprint(train_vocab.sym2freqs)
    pprint(train_vocab.sym2id)

    MIN_VOCAB_FREQ, MAX_VOCAB_CNT = 2, 10
    train_vocab = train_vocab.prune(MIN_VOCAB_FREQ, MAX_VOCAB_CNT)

    pprint(train_vocab.sym2freqs)
    pprint(train_vocab.sym2id)

    print('encode train data')
    train_data, _, _, _ = preprocessing.nlp_preprocess(train_data[0].question, train_vocab)
    print(train_data)
