# -*- coding: utf-8 -*-

import subprocess

import numpy as np
import tensorflow as tf

from jack import readers
from jack.core.data_structures import QASetting


def test_readme_fastqa():
    args = ['python3', './bin/jack-train.py', 'with', 'config=tests/test_conf/fastqa_test.yaml']
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    tf.reset_default_graph()

    fastqa_reader = readers.fastqa_reader()
    fastqa_reader.load_and_setup("tests/test_results/fastqa_reader_test")

    support = """"Architecturally, the school has a Catholic character.
    Atop the Main Building's gold dome is a golden statue of the Virgin Mary.
    Immediately in front of the Main Building and facing it, is a copper statue of
    Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the
    Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto,
    a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes,
    France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858.
    At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome),
    is a simple, modern stone statue of Mary."""

    answers = fastqa_reader([QASetting(
        question="To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
        support=[support]
    )])

    assert answers[0][0].text is not None


def test_readme_dam():
    args = ['python3', './bin/jack-train.py', 'with', 'config=tests/test_conf/dam_test.yaml']
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    tf.reset_default_graph()

    dam_reader = readers.dam_snli_reader()
    dam_reader.load_and_setup("tests/test_results/dam_reader_test")

    atomic_candidates = ['entailment', 'neutral', 'contradiction']
    answers = dam_reader([QASetting(
        question="The boy plays with the ball.",
        support=["The boy plays with the ball."],
        candidates=atomic_candidates
    )])

    assert answers[0] is not None
    assert isinstance(answers[0][0].score, np.float32)
    assert answers[0][0].text in atomic_candidates
