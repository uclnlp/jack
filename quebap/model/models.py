import tensorflow as tf
import numpy as np
import quebap.util.tfutil as tfutil
from quebap.util import tfutil as tfutil
from quebap.sisyphos.models import embedder, get_total_trainable_variables, get_total_variables, conditional_reader, predictor

def conditional_reader_model(embeddings=None, **options):
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
    targets = tf.placeholder(tf.int64, [None], "targets")

    with tf.variable_scope("embedders") as varscope:
        question_embedded = embedder(question, options["repr_dim_output"], options["vocab_size"], embeddings=embeddings)
        varscope.reuse_variables()
        support_embedded = embedder(support, options["repr_dim_output"], options["vocab_size"], embeddings=embeddings)


    print('TRAINABLE VARIABLES (only embeddings): %d'%get_total_trainable_variables())


    outputs,states = conditional_reader(question_embedded, question_lengths,
                                support_embedded, support_lengths,
                                options["repr_dim_output"])
    #states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
    output = tf.concat(1, [states[0][1], states[1][1]])

    logits, loss, predict = predictor(output, targets, options["target_size"])

    print('TRAINABLE VARIABLES (embeddings + model): %d'%get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d'%get_total_variables())


    return (logits, loss, predict), \
           {'question': question, 'question_lengths': question_lengths,
            'support': support, 'support_lengths': support_lengths,
            'targets': targets} #placeholders

