# -*- coding: utf-8 -*-
from jack.core import SharedResources
from jack.io.embeddings import load_embeddings
from jack.util.vocab import Vocab
import numpy as np


def test_readme():
    from jack import readers

    # fastqa_reader = readers.fastqa_reader()
    # fastqa_reader.setup_from_file("./fastqa_reader")
