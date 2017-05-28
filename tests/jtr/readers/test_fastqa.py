import numpy as np

import jtr.jack.readers as readers
from jtr.jack.data_structures import load_labelled_data
from jtr.jack.tasks.xqa.util import tokenize
from jtr.io.embeddings.embeddings import Embeddings
from jtr.io.embeddings.vocabulary import Vocabulary
from jtr.preprocess.vocab import Vocab
from jtr.jack.core import SharedVocabAndConfig


def test_fastqa():
    data = load_labelled_data('tests/test_data/squad/snippet_jtr.json')
    questions = []
    # fast qa must be initialized with existing embeddings, so we create some
    vocab = dict()
    for question, _ in data:
        questions.append(question)
        for t in tokenize(question.question):
            if t not in vocab:
                vocab[t] = len(vocab)
    vocab = Vocabulary(vocab)
    embeddings = Embeddings(vocab, np.random.random([len(vocab), 10]))

    # we need a vocabulary (with embeddings for our fastqa_reader, but this is not always necessary)
    vocab = Vocab(emb=embeddings, init_from_embeddings=True)

    # ... and a config
    config = {"batch_size": 1, "repr_dim": 10, "repr_dim_input": embeddings.lookup.shape[1],
              "with_char_embeddings": True}

    # create/setup reader
    shared_resources = SharedVocabAndConfig(vocab, config)
    fastqa_reader = readers.readers["fastqa_reader"](shared_resources)
    fastqa_reader.setup_from_data(data)

    answers = fastqa_reader(questions)

    assert answers, "FastQA reader should produce answers"
