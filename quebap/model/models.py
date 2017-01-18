"""
Provides models that define no placeholders ('models_np')

todo: include other models; goal: should replace models.py
"""

import tensorflow as tf
from quebap.sisyphos.models import get_total_trainable_variables, get_total_variables, conditional_reader, \
    predictor, boe_reader, bag_reader, bilstm_readers, reader



def boe_nosupport_cands_reader_model(placeholders, nvocab, **options):
    """
    Bag of embedding reader with pairs of (question, support) and candidates
    """

    # Model
    # [batch_size, max_seq1_length]
    question = placeholders['question']

    # [batch_size, candidate_size]
    targets = placeholders['targets']

    # [batch_size, max_num_cands]
    candidates = placeholders['candidates']

    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)
        varscope.reuse_variables()
        candidates_embedded = nvocab(candidates)

    print('TRAINABLE VARIABLES (only embeddings): %d' % get_total_trainable_variables())
    question_encoding = tf.reduce_sum(question_embedded, 1)

    scores = logits = tf.reduce_sum(tf.expand_dims(question_encoding, 1) * candidates_embedded, 2)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')

    print('TRAINABLE VARIABLES (embeddings + model): %d' % get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d' % get_total_variables())

    return (logits, loss, predict)



def bilstm_nosupport_reader_model_with_cands(placeholders, nvocab, **options):
    """
    LSTM reader with pairs of (question, support) and candidates
    """

    # Model
    # [batch_size, max_seq1_length]
    question = placeholders['question']
    # [batch_size]
    question_lengths = placeholders['question_lengths']

    # [batch_size, max_seq2_length]
    support = placeholders['support']

    # [batch_size, candidate_size]
    targets = placeholders['targets']

    # [batch_size, max_num_cands]
    candidates = placeholders['candidates']

    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)
        varscope.reuse_variables()
        candidates_embedded = nvocab(candidates)

    print('TRAINABLE VARIABLES (only embeddings): %d' % get_total_trainable_variables())

    #seq1, seq1_lengths, seq2, seq2_lengths, output_size, scope=None, drop_keep_prob=1.0
    with tf.variable_scope("bilstm_reader") as varscope1:
        # seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        seq1_output, seq1_states = reader(question_embedded, question_lengths, options["repr_dim_output"]/2, scope=varscope1, drop_keep_prob=options["drop_keep_prob"])

    #outputs, states
    # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
    output = tf.concat(1, [seq1_states[0][1], seq1_states[1][1]])

    scores = logits = tf.reduce_sum(tf.mul(tf.expand_dims(output, 1), candidates_embedded), 2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')


    print('TRAINABLE VARIABLES (embeddings + model): %d' % get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d' % get_total_variables())

    return (logits, loss, predict)



def bilstm_reader_model_with_cands(placeholders, nvocab, **options):
    """
    LSTM reader with pairs of (question, support) and candidates
    """

    # Model
    # [batch_size, max_seq1_length]
    question = placeholders['question']
    # [batch_size]
    question_lengths = placeholders['question_lengths']

    # [batch_size, max_seq2_length]
    support = placeholders['support']
    # [batch_size]
    support_lengths = placeholders['support_lengths']

    # [batch_size, candidate_size]
    targets = placeholders['targets']

    # [batch_size, max_num_cands]
    candidates = placeholders['candidates']

    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)
        varscope.reuse_variables()
        support_embedded = nvocab(support)
        varscope.reuse_variables()
        candidates_embedded = nvocab(candidates)

    # todo: add option for attentive reader

    print('TRAINABLE VARIABLES (only embeddings): %d' % get_total_trainable_variables())

    seq1_output, seq1_states, seq2_output, seq2_states = bilstm_readers(support_embedded, support_lengths,
                                         question_embedded, question_lengths,
                                         options["repr_dim_output"]/4, drop_keep_prob=options["drop_keep_prob"])   #making output 1/4 big so that it matches with candidates

    #outputs, states
    # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
    output = tf.concat(1, [seq1_states[0][1], seq1_states[1][1], seq2_states[0][1], seq2_states[1][1]])


    scores = logits = tf.reduce_sum(tf.mul(tf.expand_dims(output, 1), candidates_embedded), 2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')


    print('TRAINABLE VARIABLES (embeddings + model): %d' % get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d' % get_total_variables())

    return (logits, loss, predict)



def conditional_reader_model_with_cands(placeholders, nvocab, **options):
    """
    Bidirectional conditional reader with pairs of (question, support) and candidates
    """

    # Model
    # [batch_size, max_seq1_length]
    question = placeholders['question']
    # [batch_size]
    question_lengths = placeholders['question_lengths']

    # [batch_size, max_seq2_length]
    support = placeholders['support']
    # [batch_size]
    support_lengths = placeholders['support_lengths']

    # [batch_size, candidate_size]
    targets = placeholders['targets']

    # [batch_size, max_num_cands]
    candidates = placeholders['candidates']

    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)
        varscope.reuse_variables()
        support_embedded = nvocab(support)
        varscope.reuse_variables()
        candidates_embedded = nvocab(candidates)

    # todo: add option for attentive reader

    print('TRAINABLE VARIABLES (only embeddings): %d' % get_total_trainable_variables())

    # outputs,states = conditional_reader(question_embedded, question_lengths,
    #                            support_embedded, support_lengths,
    #                            options["repr_dim_output"])
    # todo: verify validity of exchanging question and support. Below: encode question, conditioned on support encoding.
    outputs, states = conditional_reader(support_embedded, support_lengths,
                                         question_embedded, question_lengths,
                                         options["repr_dim_output"]/2, drop_keep_prob=options["drop_keep_prob"])   #making output half as big so that it matches with candidates
    # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
    output = tf.concat(1, [states[0][1], states[1][1]])


    scores = logits = tf.reduce_sum(tf.mul(tf.expand_dims(output, 1), candidates_embedded), 2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')


    print('TRAINABLE VARIABLES (embeddings + model): %d' % get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d' % get_total_variables())

    return (logits, loss, predict)


def boe_support_cands_reader_model(placeholders, nvocab, **options):
    """
    Bag of embedding reader with pairs of (question, support) and candidates
    """

    # Model
    # [batch_size, max_seq1_length]
    question = placeholders['question']
    # [batch_size, max_seq2_length]
    support = placeholders['support']

    # [batch_size, candidate_size]
    targets = placeholders['targets']

    # [batch_size, max_num_cands]
    candidates = placeholders['candidates']

    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)
        varscope.reuse_variables()
        support_embedded = nvocab(support)
        varscope.reuse_variables()
        candidates_embedded = nvocab(candidates)

    # todo: add option for attentive reader

    print('TRAINABLE VARIABLES (only embeddings): %d' % get_total_trainable_variables())

    question_encoding = tf.expand_dims(tf.reduce_sum(question_embedded, 1, keep_dims=True), 3)

    candidate_encoding = tf.expand_dims(candidates_embedded, 2)
    support_encoding = tf.expand_dims(tf.reduce_sum(support_embedded, 1, keep_dims=True), 3)

    q_c = question_encoding * candidate_encoding
    combined = q_c * support_encoding
    #combined = question_encoding * candidate_encoding * support_encoding

    # [batch_size, dim]
    scores = logits = tf.reduce_sum(combined, (2, 3))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')

    print('TRAINABLE VARIABLES (embeddings + model): %d' % get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d' % get_total_variables())

    return (logits, loss, predict)


def boenosupport_reader_model(placeholders, nvocab, **options):
    """
    Bag of embeddings reader with questions
    """

    # Model
    # [batch_size, max_seq1_length]
    question = placeholders['question'] #tf.placeholder(tf.int64, [None, None], "question")
    # [batch_size]
    question_lengths = placeholders['question_lengths'] #tf.placeholder(tf.int64, [None], "question_lengths")

    # [batch_size]
    targets = placeholders['answers'] #tf.placeholder(tf.int64, [None], "answers")

    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)

    print('TRAINABLE VARIABLES (only embeddings): %d' % get_total_trainable_variables())

    output = bag_reader(question_embedded, question_lengths)
    print("INPUT SHAPE " + str(question_embedded.get_shape()))
    print("OUTPUT SHAPE " + str(output.get_shape()))

    logits, loss, predict = predictor(output, targets, options["answer_size"])

    print('TRAINABLE VARIABLES (embeddings + model): %d' % get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d' % get_total_variables())

    return (logits, loss, predict)



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

    return (logits, loss, predict)




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
