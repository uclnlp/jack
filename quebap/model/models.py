import tensorflow as tf
import numpy as np
import quebap.util.tfutil as tfutil
from quebap.util import tfutil as tfutil
from quebap.sisyphos.models import get_total_trainable_variables, get_total_variables, conditional_reader, boe_reader, \
    predictor


# class ReaderModel():

def model(self, supports, questions, candidates, answers):
    """
    Validator for each model. This allows to define properties that a model supports. For each attribute,
    either pass a list (model supports multiple options) or a string (only one option supported)

    Args:
        `supports`:
            "none" (the model does not use supporting sentences) ,
            "single" (one supporting sentence per instance),
            "multiple" (multiple supporting sentences per instance)
        `questions`:
            "single",
            "multiple" (same as for supports)
        `candidates`:
            "open" (the model does not use candidates, it either generates them itself or uses some other mechanism,
               e.g. predicting labels on token-level),
            "per-instance" (different candidates for each instance, this is typical for question-answering problems),
            "fixed" (same candidates for all instances, this is typical for RTE, sentiment analysis, universal schema knowledge base population setup)
        `answers`
            "single" (only one possible answer taken into consideration),
            "multiple" (multiple answers possible)

    Returns:
        ReaderModel instance
    """
    # Allow passing of single strings.
    if isinstance(supports, str):
        supports = (supports,)
    if isinstance(questions, str):
        questions = (questions,)
    if isinstance(candidates, str):
        candidates = (candidates,)
    if isinstance(answers, str):
        answers = (answers,)
    supports = set(supports)
    questions = set(questions)
    candidates = set(candidates)
    answers = set(answers)
    # "Type" checking.
    supp_alts = {'none', 'single', 'multiple'}
    if supports - supp_alts:
        raise TypeError('invalid supports argument(s): {}'.format(
            ', '.join(supports - supp_alts)))
    q_alts = {'single', 'multiple'}
    if questions - q_alts:
        raise TypeError('invalid questions argument(s): {}'.format(
            ', '.join(questions - q_alts)))
    can_alts = {'open', 'per-instance', 'fixed'}
    if candidates - can_alts:
        raise TypeError('invalid candidates argument(s): {}'.format(
            ', '.join(candidates - can_alts)))
    ans_alts = {'single', 'multiple'}
    if answers - ans_alts:
        raise TypeError('invalid answers argument(s): {}'.format(
            ', '.join(answers - ans_alts)))
    self.supports = supports
    self.questions = questions
    self.candidates = candidates
    self.answers = answers
    return self


# @model(supports="single", questions="single", candidates="fixed", answers="single") #decorator
def conditional_reader_model(nvocab, **options):
    """
    Bidirectional conditional reader with pairs of (question, support)
    """

    # Model
    # [batch_size, max_seq1_length]
    question = tf.placeholder(tf.int64, [None, None], "question")
    # [batch_size]
    question_lengths = tf.placeholder(tf.int64, [None], "question_lengths")

    # [batch_size, max_seq2_length]
    support = tf.placeholder(tf.int64, [None, None], "support")
    # [batch_size]
    support_lengths = tf.placeholder(tf.int64, [None], "support_lengths")

    # [batch_size]
    targets = tf.placeholder(tf.int64, [None], "answers")

    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)
        varscope.reuse_variables()
        support_embedded = nvocab(support)

    # todo: add option for attentive reader

    print('TRAINABLE VARIABLES (only embeddings): %d' % get_total_trainable_variables())

    # outputs,states = conditional_reader(question_embedded, question_lengths,
    #                            support_embedded, support_lengths,
    #                            options["repr_dim_output"])
    # todo: verify validity of exchanging question and support. Below: encode question, conditioned on support encoding.
    outputs, states = conditional_reader(support_embedded, support_lengths,
                                         question_embedded, question_lengths,
                                         options["repr_dim_output"])
    # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
    output = tf.concat(1, [states[0][1], states[1][1]])
    # todo: extend

    logits, loss, predict = predictor(output, targets, options["answer_size"])

    print('TRAINABLE VARIABLES (embeddings + model): %d' % get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d' % get_total_variables())

    return (logits, loss, predict), \
           {'question': question, 'question_lengths': question_lengths,
            'support': support, 'support_lengths': support_lengths,
            'answers': targets}  # placeholders


# @model(supports="single", questions="single", candidates="fixed", answers="single") #decorator
def boe_reader_model(nvocab, **options):
    """
    Bag of embeddings reader with pairs of (question, support)
    """

    # Model
    # [batch_size, max_seq1_length]
    question = tf.placeholder(tf.int64, [None, None], "question")
    # [batch_size]
    question_lengths = tf.placeholder(tf.int64, [None], "question_lengths")

    # [batch_size, max_seq2_length]
    support = tf.placeholder(tf.int64, [None, None], "support")
    # [batch_size]
    support_lengths = tf.placeholder(tf.int64, [None], "support_lengths")

    # [batch_size]
    targets = tf.placeholder(tf.int64, [None], "answers")

    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)
        varscope.reuse_variables()
        support_embedded = nvocab(support)

    print('TRAINABLE VARIABLES (only embeddings): %d' % get_total_trainable_variables())

    output = boe_reader(question_embedded, question_lengths,
                        support_embedded, support_lengths)
    print("INPUT SHAPE " + str(question_embedded.get_shape()))
    print("OUTPUT SHAPE " + str(output.get_shape()))

    logits, loss, predict = predictor(output, targets, options["answer_size"])

    print('TRAINABLE VARIABLES (embeddings + model): %d' % get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d' % get_total_variables())

    return (logits, loss, predict), \
           {'question': question, 'question_lengths': question_lengths,
            'support': support, 'support_lengths': support_lengths,
            'answers': targets}  # placeholders
