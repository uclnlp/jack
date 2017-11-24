import os
from os.path import join

"""
Global config options
"""

TRIVIA_QA = os.environ.get('TRIVIAQA_HOME', None)
TRIVIA_QA_UNFILTERED = os.environ.get('TRIVIAQA_UNFILTERED_HOME', None)

CORPUS_DIR = join(os.environ.get('TRIVIAQA_HOME', ''), "preprocessed")

VEC_DIR = ''
