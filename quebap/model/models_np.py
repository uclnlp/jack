"""
Provides models that define no placeholders ('models_np')

todo: include other models; goal: should replace models.py
"""

import tensorflow as tf
from quebap.sisyphos.models import get_total_trainable_variables, get_total_variables, conditional_reader, \
    predictor, boe_reader



def boe_reader_model(placeholders, nvocab, **options):
    """
    Bag of embeddings reader with pairs of (question, support)
    """

    # [batch_size, max_seq1_length]
    question = placeholders['question']
    # [batch_size]
    question_lengths = placeholders["question_lengths"]
    # [batch_size, max_seq2_length]
    support = placeholders["support"]
    # [batch_size]
    support_lengths = placeholders["support_lengths"]
    # [batch_size]
    targets = placeholders["answers"]

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

    return logits, loss, predict




def conditional_reader_model(placeholders, nvocab, **options):
    """
    Bidirectional conditional reader with pairs of (question, support)
    placeholders: dictionary that should contain placeholders for at least the following keys:
    "question"
    "question_length"
    "support"
    "support_length"
    "answers"
    """

    # [batch_size, max_seq1_length]
    question = placeholders['question']
    # [batch_size]
    question_lengths = placeholders["question_lengths"]
    # [batch_size, max_seq2_length]
    support = placeholders["support"]
    # [batch_size]
    support_lengths = placeholders["support_lengths"]
    # [batch_size]
    targets = placeholders["answers"]

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
                                         options["repr_dim_output"], drop_keep_prob=options["drop_keep_prob"])
    # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
    output = tf.concat(1, [states[0][1], states[1][1]])
    # todo: extend

    logits, loss, predict = predictor(output, targets, options["answer_size"])

    print('TRAINABLE VARIABLES (embeddings + model): %d' % get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d' % get_total_variables())

    return (logits, loss, predict)
