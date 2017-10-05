# -*- coding: utf-8 -*-

import subprocess

import tensorflow as tf

from jack import readers


def test_readme():
    args = ['python3', 'jack/train_reader.py', 'with', 'config=tests/test_conf/fastqa_test.yaml']
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    for line in str(err).split('\\n'):
        if 'Iter 1' in line:
            assert 'f1: 0.106' in line
        if 'Iter 2' in line:
            assert 'f1: 0.077' in line
        if 'Iter 3' in line:
            assert 'f1: 0.112' in line
        if 'Iter 4' in line:
            assert 'f1: 0.113' in line
        if 'Iter 5' in line:
            assert 'f1: 0.146' in line

    tf.reset_default_graph()

    fastqa_reader = readers.fastqa_reader()
    fastqa_reader.load_and_setup("tests/test_results/fastqa_reader_test")
