# -*- coding: utf-8 -*-

"""
Provides models that define no placeholders ('models_np')

todo: include other models; goal: should replace models.py
"""

import tensorflow as tf
import sys

import logging
logger = logging.getLogger(__name__)


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

    logger.info('TRAINABLE VARIABLES (only embeddings): {}'.format(get_total_trainable_variables()))
    question_encoding = tf.reduce_sum(question_embedded, 1)

    scores = logits = tf.reduce_sum(tf.expand_dims(question_encoding, 1) * candidates_embedded, 2)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores,
        labels=targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')

    logger.info('TRAINABLE VARIABLES (embeddings + model): {}'.format(get_total_trainable_variables()))
    logger.info('ALL VARIABLES (embeddings + model): {}'.format(get_total_variables()))

    return logits, loss, predict


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

    logger.info('TRAINABLE VARIABLES (only embeddings): {}'.format(get_total_trainable_variables()))

    #seq1, seq1_lengths, seq2, seq2_lengths, output_size, scope=None, drop_keep_prob=1.0
    with tf.variable_scope("bilstm_reader") as varscope1:
        # seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        seq1_output, seq1_states = reader(question_embedded, question_lengths, options["repr_dim_output"]/2, scope=varscope1, drop_keep_prob=options["drop_keep_prob"])

    #outputs, states
    # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
    output = tf.concat([seq1_states[0][1], seq1_states[1][1]], 1)

    scores = logits = tf.reduce_sum(tf.mul(tf.expand_dims(output, 1), candidates_embedded), 2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores,
        labels=targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')

    logger.info('TRAINABLE VARIABLES (embeddings + model): {}'.format(get_total_trainable_variables()))
    logger.info('ALL VARIABLES (embeddings + model): {}'.format(get_total_variables()))

    return logits, loss, predict


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

    logger.info('TRAINABLE VARIABLES (only embeddings): {}'.format(get_total_trainable_variables()))

    seq1_output, seq1_states, seq2_output, seq2_states = bilstm_readers(support_embedded, support_lengths,
                                         question_embedded, question_lengths,
                                         options["repr_dim_output"]/4, drop_keep_prob=options["drop_keep_prob"])   #making output 1/4 big so that it matches with candidates

    #outputs, states
    # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
    output = tf.concat([seq1_states[0][1], seq1_states[1][1],
        seq2_states[0][1], seq2_states[1][1]], 1)

    scores = logits = tf.reduce_sum(tf.mul(tf.expand_dims(output, 1), candidates_embedded), 2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores,
        labels=targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')

    logger.info('TRAINABLE VARIABLES (embeddings + model): {}'.format(get_total_trainable_variables()))
    logger.info('ALL VARIABLES (embeddings + model): {}'.format(get_total_variables()))

    return logits, loss, predict


def boe_multisupport_avg_reader_with_cands(placeholders, nvocab, **options):
    """
    For datasets with multiple supports: bow reader with pairs of (question, [support]) and candidates. Mean averages the support embeddings.
    """

    # Model
    # [batch_size, max_seq1_length]
    question = placeholders['question']

    # [batch_size, num_sup, max_seq2_length]
    support = placeholders['support']

    # [batch_size, candidate_size]
    targets = tf.to_float(placeholders['targets'])

    # [batch_size, max_num_cands]
    candidates = placeholders['candidates']

    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)
        varscope.reuse_variables()
        support_embedded = nvocab(support)
        varscope.reuse_variables()
        candidates_embedded = nvocab(candidates)

    # support encodings are mean averaged
    support_embedded = tf.reduce_mean(support_embedded, 1)

    logger.info('TRAINABLE VARIABLES (only embeddings): {}'.format(get_total_trainable_variables()))

    question_embedded = tf.reduce_sum(question_embedded, 1)
    support_embedded = tf.reduce_sum(support_embedded, 1)

    # batch matrix multiplication to get per-candidate scores
    scores = tf.einsum('bid,bcd->bc', tf.expand_dims(question_embedded + support_embedded, 1), candidates_embedded)

    loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores,
        labels=targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(scores), 1, name='prediction')

    logger.info('TRAINABLE VARIABLES (embeddings + model): {}'.format(get_total_trainable_variables()))
    logger.info('ALL VARIABLES (embeddings + model): {}'.format(get_total_variables()))

    return scores, loss, predict




def boe_multisupport_lossavg_reader_with_cands(placeholders, nvocab, **options):
    """
    For datasets with multiple supports: bow reader with pairs of (question, [support]) and candidates.
    Pairs each support encoding with the repective question and candidate encodings and
    """

    # Model
    # [batch_size, max_seq1_length]
    question = placeholders['question']

    # [batch_size, num_sup, max_seq2_length]
    support = placeholders['support']

    # [batch_size, candidate_size]
    targets = tf.to_float(placeholders['targets'])

    # [batch_size, max_num_cands]
    candidates = placeholders['candidates']

    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)
        varscope.reuse_variables()
        support_embedded = nvocab(support)
        varscope.reuse_variables()
        candidates_embedded = nvocab(candidates)

    print('TRAINABLE VARIABLES (only embeddings): %d' % get_total_trainable_variables())

    dim1s, dim2s, dim3s, dim4s = tf.unstack(tf.shape(support_embedded))  # [batch_size, num_supports, max_seq2_length, emb_dim]

    # iterate through all supports
    num_steps = dim2s

    initial_outputs = tf.TensorArray(size=num_steps, dtype='float32')
    initial_i = tf.constant(0, dtype='int32')

    def should_continue(i, *args):
        # execute the loop for all i supports
        return i < num_steps

    def iteration(i, outputs_):
        # get all instances, take only i-th support, flatten so this becomes a 3-dim tensor
        sup_batchi = tf.reshape(tf.slice(support_embedded, [0, i, 0, 0], [dim1s, 1, dim3s, dim4s]),
                                [dim1s, dim3s, dim4s])  # [batch_size, num_supports, max_seq2_length]

        question_encoding = tf.reduce_sum(question_embedded, 1)
        support_encoding = tf.reduce_sum(sup_batchi, 1)

        # batch matrix multiplication to get per-candidate scores
        scores = tf.einsum('bid,bcd->bc', tf.expand_dims(question_encoding + support_encoding, 1), candidates_embedded)

        # append scores for the i-th support to previous supports so we can combine scores for all supports later
        outputs_ = outputs_.write(i, scores)

        return i + 1, outputs_

    i, outputs = tf.while_loop(
        should_continue, iteration,
        [initial_i, initial_outputs])

    # packs along axis 0
    outputs_logits = outputs.pack()  # [num_support, batch_size, num_cands]

    # so we'll need to combine losses on axis 0 as well
    # here: plug in different ways of combining losses
    scores = tf.reduce_sum(outputs_logits, 0)  # [batch_size, num_cands]

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores,
        labels=targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(scores), 1, name='prediction')

    print('TRAINABLE VARIABLES (embeddings + model): %d' % get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d' % get_total_variables())

    return (scores, loss, predict)





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

    logger.info('TRAINABLE VARIABLES (only embeddings): {}'.format(get_total_trainable_variables()))

    # outputs,states = conditional_reader(question_embedded, question_lengths,
    #                            support_embedded, support_lengths,
    #                            options["repr_dim_output"])
    # todo: verify validity of exchanging question and support. Below: encode question, conditioned on support encoding.
    outputs, states = conditional_reader(support_embedded, support_lengths,
                                         question_embedded, question_lengths,
                                         options["repr_dim_output"]/2, drop_keep_prob=options["drop_keep_prob"])   #making output half as big so that it matches with candidates
    # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
    output = tf.concat([states[0][1], states[1][1]], 1)

    scores = logits = tf.reduce_sum(tf.mul(tf.expand_dims(output, 1), candidates_embedded), 2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores,
        labels=targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')

    logger.info('TRAINABLE VARIABLES (embeddings + model): {}'.format(get_total_trainable_variables()))
    logger.info('ALL VARIABLES (embeddings + model): {}'.format(get_total_variables()))

    return logits, loss, predict


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

    logger.info('TRAINABLE VARIABLES (only embeddings): {}'.format(get_total_trainable_variables()))

    question_encoding = tf.expand_dims(tf.reduce_sum(question_embedded, 1, keep_dims=True), 3)

    candidate_encoding = tf.expand_dims(candidates_embedded, 2)
    support_encoding = tf.expand_dims(tf.reduce_sum(support_embedded, 1, keep_dims=True), 3)

    q_c = question_encoding * candidate_encoding
    combined = q_c * support_encoding
    # combined = question_encoding * candidate_encoding * support_encoding

    # [batch_size, dim]
    scores = logits = tf.reduce_sum(combined, (2, 3))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores,
        labels=targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')

    logger.info('TRAINABLE VARIABLES (embeddings + model): {}'.format(get_total_trainable_variables()))
    logger.info('ALL VARIABLES (embeddings + model): {}'.format(get_total_variables()))

    return logits, loss, predict


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

    logger.info('TRAINABLE VARIABLES (only embeddings): {}'.format(get_total_trainable_variables()))

    output = bag_reader(question_embedded, question_lengths)
    logger.info("INPUT SHAPE {}".format(str(question_embedded.get_shape())))
    logger.info("OUTPUT SHAPE {}".format(str(output.get_shape())))

    logits, loss, predict = predictor(output, targets, options["answer_size"])

    logger.info('TRAINABLE VARIABLES (embeddings + model): {}'.format(get_total_trainable_variables()))
    logger.info('ALL VARIABLES (embeddings + model): {}'.format(get_total_variables()))

    return logits, loss, predict


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

    logger.info('TRAINABLE VARIABLES (only embeddings): {}'.format(get_total_trainable_variables()))

    output = boe_reader(question_embedded, question_lengths, support_embedded, support_lengths)
    logger.info("INPUT SHAPE {}".format(str(question_embedded.get_shape())))
    logger.info("OUTPUT SHAPE {}".format(str(output.get_shape())))

    logits, loss, predict = predictor(output, targets, options["answer_size"])

    logger.info('TRAINABLE VARIABLES (embeddings + model): {}'.format(get_total_trainable_variables()))
    logger.info('ALL VARIABLES (embeddings + model): {}'.format(get_total_variables()))

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

    logger.info('TRAINABLE VARIABLES (only embeddings): {}'.format(get_total_trainable_variables()))

    # outputs,states = conditional_reader(question_embedded, question_lengths,
    #                            support_embedded, support_lengths,
    #                            options["repr_dim_output"])
    # todo: verify validity of exchanging question and support. Below: encode question, conditioned on support encoding.
    outputs, states = conditional_reader(support_embedded, support_lengths,
                                         question_embedded, question_lengths,
                                         options["repr_dim_output"], drop_keep_prob=options["drop_keep_prob"])
    # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
    output = tf.concat([states[0][1], states[1][1]], 1)
    # todo: extend

    logits, loss, predict = predictor(output, targets, options["answer_size"])

    logger.info('TRAINABLE VARIABLES (embeddings + model): {}'.format(get_total_trainable_variables()))
    logger.info('ALL VARIABLES (embeddings + model): {}'.format(get_total_variables()))

    return logits, loss, predict


def get_total_trainable_variables():
    """Calculates and returns the number of trainable parameters in the model."""
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def get_total_variables():
    """Calculates and returns the number of parameters in the model (these can be fixed)."""
    total_parameters = 0
    for variable in tf.global_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def reader(inputs, lengths, output_size, contexts=(None, None), scope=None, drop_keep_prob=1.0):
    """Dynamic bi-LSTM reader; can be conditioned with initial state of other rnn.

    Args:
        inputs (tensor): The inputs into the bi-LSTM
        lengths (tensor): The lengths of the sequences
        output_size (int): Size of the LSTM state of the reader.
        context (tensor=None, tensor=None): Tuple of initial (forward, backward) states
                                  for the LSTM
        scope (string): The TensorFlow scope for the reader.
        drop_keep_drop (float=1.0): The keep probability for dropout.

    Returns:
        Outputs (tensor): The outputs from the bi-LSTM.
        States (tensor): The cell states from the bi-LSTM.
    """
    with tf.variable_scope(scope or "reader") as varscope:
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        #if drop_keep_prob != 1.0:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=drop_keep_prob, input_keep_prob=drop_keep_prob, seed=1233)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            inputs,
            sequence_length=lengths,
            initial_state_fw=contexts[0],
            initial_state_bw=contexts[1],
            dtype=tf.float32
        )

        # ( (outputs_fw,outputs_bw) , (output_state_fw,output_state_bw) )
        # in case LSTMCell: output_state_fw = (c_fw,h_fw), and output_state_bw = (c_bw,h_bw)
        # each [batch_size x max_seq_length x output_size]
        return outputs, states


def conditional_reader(seq1, seq1_lengths, seq2, seq2_lengths, output_size, scope=None, drop_keep_prob=1.0):
    """Duo of bi-LSTMs over seq1 and seq2 where seq2 is conditioned on by the final state of seq1.

    See also: conditional_reader_model: Instantiates either condition or
    attentive reader with placeholders.
    Args:
        seq1 (tensor): The inputs into the first bi-LSTM.
        seq1_lengths (tensor): The lengths of the sequences.
        seq2 (tensor): The inputs into the second bi-LSTM.
        seq1_lengths (tensor): The lengths of the sequences.
        output_size (int): Size of the LSTMs state.
        scope (string): The TensorFlow scope for the reader.
        drop_keep_drop (float=1.0): The keep propability for dropout.

    Returns:
        Outputs (tensor): The outputs from the second bi-LSTM.
        States (tensor): The cell states from the second bi-LSTM.
    """
    with tf.variable_scope(scope or "conditional_reader_seq1") as varscope1:
        #seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        _, seq1_states = reader(seq1, seq1_lengths, output_size, scope=varscope1, drop_keep_prob=drop_keep_prob)
    with tf.variable_scope(scope or "conditional_reader_seq2") as varscope2:
        varscope1.reuse_variables()
        # each [batch_size x max_seq_length x output_size]
        return reader(seq2, seq2_lengths, output_size, seq1_states, scope=varscope2, drop_keep_prob=drop_keep_prob)


def bilstm_readers(seq1, seq1_lengths, seq2, seq2_lengths, output_size, scope=None, drop_keep_prob=1.0):
    """Duo of independent bi-LSTMs over seq1 and seq2.

    Args:
        seq1 (tensor): The inputs into the first bi-LSTM.
        seq1_lengths (tensor): The lengths of the sequences.
        seq2 (tensor): The inputs into the second bi-LSTM.
        seq1_lengths (tensor): The lengths of the sequences.
        output_size (int): Size of the LSTMs state.
        scope (string): The TensorFlow scope for the reader.
        drop_keep_drop (float=1.0): The keep probability for dropout.

    Returns:
        Outputs (tensor): The outputs from the second bi-LSTM.
        States (tensor): The cell states from the second bi-LSTM.
    """
    # same as conditional_reader, apart from that second lstm is initialised randomly
    with tf.variable_scope(scope or "bilstm_reader_seq1") as varscope1:
        # seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        seq1_output, seq1_states = reader(seq1, seq1_lengths, output_size, scope=varscope1, drop_keep_prob=drop_keep_prob)
    with tf.variable_scope(scope or "bilstm_reader_seq2") as varscope2:
        varscope1.reuse_variables()
        # each [batch_size x max_seq_length x output_size]
        seq2_output, seq2_states = reader(seq2, seq2_lengths, output_size, scope=varscope2, drop_keep_prob=drop_keep_prob)
    return seq1_output, seq1_states, seq2_output, seq2_states


def bag_reader(inputs, lengths):
    """Sums along the feature dimension of inputs. Returns sum per sample.

    Args:
        inputs (tensor): Input sequence.
        lengths (tensor):  Variable lengths of the input sequence.
    Returns:
        output (tensor: Sum per sample.
    """
    output=tf.reduce_sum(inputs,1,keep_dims=False)
    return output


def boe_reader(seq1, seq1_lengths, seq2, seq2_lengths):
    """Sums the feature dimension of two sequences and return its concatenation

    Args:
        seq1 (tensor): First input sequence.
        seq1_lengths (tensor):  Variable lengths of the first input sequence.
        seq2 (tensor): Second input sequence.
        seq2_lengths (tensor):  Variable lengths of the second input sequence.
    Returns:
        output (tensor: Concatenation of the sums per sample for the sequences.
    """
    output1 = bag_reader(seq1, seq1_lengths)
    output2 = bag_reader(seq2, seq2_lengths)
    # each [batch_size x max_seq_length x output_size]
    return tf.concat([output1,output2], 1)


def predictor(inputs, targets, target_size):
    """Projects inputs onto targets. Returns logits, loss, argmax.

    Creates fully connected projection layer(logits). Then applies cross entropy
    softmax to get the loss. Calculate predictions via argmax.
    Args:
        inputs (tensor): Input into the projection layer.
        targets (tensor): Targets for the loss function.
        target_size (int): Size of the targets (used in projection layer).
    """
    init = tf.contrib.layers.xavier_initializer(uniform=True) #uniform=False for truncated normal
    logits = tf.contrib.layers.fully_connected(inputs, target_size, weights_initializer=init, activation_fn=None)
    #note: standard relu applied; switch off at last layer with activation_fn=None!

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
            labels=targets), name='predictor_loss')
    predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')
    return logits, loss, predict


def attentive_reader_model(output_size, target_size, nvocab, attentive=False):
    """Creates reader that is attentive (each step) or conditional (last step)

    Reference: Teaching Machines to Read and Comprehend
    Link: https://arxiv.org/pdf/1506.03340.pdf

    Creates either a conditional reader (condition on last timestep only) or
    attentive reader (condition on all timesteps).
    The model consists of two bi-directional LSTM where the second LSTM
    is conditioned by the last cell state (conditional) or by an attention
    value from all past cell states.

    Args:
        output_size (int): Size of the two bi-LSTMs.
        target_size (int): Size of the targets (number of labels).
        nvocab (NeuralVocab): NeuralVocab class; maps word-ids to embeddings.
        attentive (bool=False): To create the attentive reader or not.
    Returns:
        (logits (tensor), loss (tensor), predict (tensor): Triple of logits,
        loss and predict (argmax) tensors,
       {'sentence1': sentence1 (TensorFlow placeholder),
        'sentence1_lengths': sentence1_lengths (tensor),
        'sentence2': sentence2 (TensorFlow placeholder),
        'sentence2_lengths': sentence2_lengths (tensor),
        'targets': targets (tensor)}: Dictionary of placeholders to feed data
        into.
    """
    # Model
    # [batch_size, max_seq1_length]
    sentence1 = tf.placeholder(tf.int64, [None, None], "sentence1")
    # [batch_size]
    sentence1_lengths = tf.placeholder(tf.int64, [None], "sentence1_lengths")

    # [batch_size, max_seq2_length]
    sentence2 = tf.placeholder(tf.int64, [None, None], "sentence2")
    # [batch_size]
    sentence2_lengths = tf.placeholder(tf.int64, [None], "sentence2_lengths")

    # [batch_size]
    targets = tf.placeholder(tf.int64, [None], "targets")

    with tf.variable_scope("embedders") as varscope:
        seq1_embedded = nvocab(sentence1)
        varscope.reuse_variables()
        seq2_embedded = nvocab(sentence2)
#        seq1_embedded = embedder(sentence1, input_size, vocab_size, embeddings=embeddings)
#        seq2_embedded = embedder(sentence2, input_size, vocab_size, embeddings=embeddings)


    print('TRAINABLE VARIABLES (only embeddings): %d'%get_total_trainable_variables())

    if attentive:
        le_conditional_reader = conditional_attentive_reader
    else:
        le_conditional_reader = conditional_reader

    outputs, states = le_conditional_reader(seq1_embedded, sentence1_lengths,
                                            seq2_embedded, sentence2_lengths,
                                            output_size)

    # fixme: also use bidirectional encoding for attention model?
    if isinstance(states, tf.nn.rnn_cell.LSTMStateTuple):
        output = states.h
    elif isinstance(states, tuple):
        # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )
        output = tf.concat([states[0][1], states[1][1]], 1)
    else:
        raise AttributeError

    logits, loss, predict = predictor(output, targets, target_size)

    print('TRAINABLE VARIABLES (embeddings + model): %d'%get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d'%get_total_variables())


    return (logits, loss, predict), \
           {'sentence1': sentence1, 'sentence1_lengths': sentence1_lengths,
            'sentence2': sentence2, 'sentence2_lengths': sentence2_lengths,
            'targets': targets} #placeholders


def conditional_attentive_reader(seq1, seq1_lengths, seq2, seq2_lengths,
                                 output_size, scope=None):
    """Creates attentive reader where two bi-LSTMs attend to each others state.

    Reference: Teaching Machines to Read and Comprehend
    Link: https://arxiv.org/pdf/1506.03340.pdf

    Creates an attentive reader which consists of two bi-directional LSTM
    where the second LSTM is conditioned by an attention value from all
    past cell states.

    Args:
        seq1 (tensor): Size of first input sequence.
        seq1_lengths (tensor)`: Lengths of first input sequences.
        seq2 (tensor): Size of second input sequence.
        seq2_lengths (tensor): Lengths of second input sequences.
        output_size (int): Hidden unit size of the two bi-LSTMs. 
        scope (string): The TensorFlow scope for the reader.
    Returns:
        (logits (tensor), loss (tensor), predict (tensor): Triple of logits,
        loss and predict (argmax) tensors,
       {'sentence1': sentence1 (TensorFlow placeholder),
        'sentence1_lengths': sentence1_lengths (tensor),
        'sentence2': sentence2 (TensorFlow placeholder),
        'sentence2_lengths': sentence2_lengths (tensor),
        'targets': targets (tensor)}: Dictionary of placeholders to feed data
        into.
    """
    with tf.variable_scope(scope or "conditional_attentive_reader_seq1") as varscope1:
        #seq1_states: (c_fw, h_fw), (c_bw, h_bw)
        attention_states, seq1_states = reader(seq1, seq1_lengths, output_size, scope=varscope1)
    with tf.variable_scope(scope or "conditional_attentitve_reader_seq2") as varscope2:
        varscope1.reuse_variables()

        batch_size = tf.shape(seq2)[0]
        max_time = tf.shape(seq2)[1]
        input_depth = int(seq2.get_shape()[2])

        # transforming seq2 to time major
        seq2 = tf.transpose(seq2, [1, 0, 2])
        num_units = output_size

        # fixme: very hacky and costly way
        seq2_lengths = tf.cast(seq2_lengths, tf.int32)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unpack(seq2)

        cell = tf.nn.rnn_cell.LSTMCell(num_units)

        attention_states_fw, attention_states_bw = tf.split(attention_states, 2, 0)
        attention_states = tf.concat(3, [attention_states_fw, attention_states_bw])
        attention_states = tf.squeeze(attention_states, [0])
        # transforming attention states time major
        attention_states = tf.transpose(attention_states, [1, 0, 2])

        attention_states = tf.contrib.layers.linear(attention_states, num_units)

        att_len = tf.shape(attention_states)[0]

        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:  # time == 0
                next_cell_state = cell.zero_state(batch_size, tf.float32)
            else:
                next_cell_state = cell_state
            elements_finished = (time >= seq2_lengths)

            c, query = next_cell_state

            # [att_len x batch_size x num_units]
            query_expanded = tf.tile(tf.expand_dims(query, 0), [att_len, 1, 1])

            attention_states_projected = \
                tf.contrib.layers.linear(attention_states, num_units)

            query_projected = \
                tf.contrib.layers.linear(query_expanded, num_units)

            # [att_len x batch_size x num_units]
            M = tf.tanh(attention_states_projected + query_projected)

            # [batch_size x att_len]
            logits = tf.transpose(tf.squeeze(tf.contrib.layers.linear(M, 1)))

            # [att_len x batch_size]
            alpha = tf.transpose(tf.nn.softmax(logits))

            attention_states_flat = tf.reshape(attention_states, [-1, num_units])

            alpha_flat = tf.reshape(alpha, [-1, 1])

            # [batch_size x num_units]
            r = attention_states_flat * alpha_flat
            r_reshaped = tf.reduce_sum(
                tf.reshape(r, [att_len, batch_size, num_units]), [0])

            # [batch_size x num_units]
            h = tf.tanh(tf.contrib.layers.linear(
                tf.concat(1, [query, r_reshaped]), num_units))

            next_cell_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
                lambda: inputs_ta.read(time))
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state,
                    emit_output, next_loop_state)

        outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        outputs = outputs_ta.pack()

        outputs_batch_major = tf.transpose(outputs, [1, 0, 2])

        # each [batch_size x max_seq_length x output_size]
        return outputs_batch_major, final_state



def model_f_reader_model(placeholders, nvocab, candvocab=None, **options):
    """
    modelf reader
    """
    
    # Model
    # [batch_size, 1]
    question = placeholders["question"]
    # [batch_size,1+negsamples]
    candidates = placeholders["candidates"]
    # [batch_size,1]
    answers = placeholders["answers"]
    
    with tf.variable_scope("embedders") as varscope:
        question_embedded = nvocab(question)
        candidates_embedded = candvocab(candidates)
        answers_embedded = tf.expand_dims(candvocab(answers),1)
    
    logger.info('TRAINABLE VARIABLES (only embeddings): {}'.format(get_total_trainable_variables()))
    
    logits, loss, predict = model_f_predictor(question_embedded, candidates_embedded, candidates, answers_embedded)
    
    logger.info('TRAINABLE VARIABLES (embeddings + model): {}'.format(get_total_trainable_variables()))
    logger.info('ALL VARIABLES (embeddings + model): {}'.format(get_total_variables()))
    
    return logits, loss, predict



def model_f_predictor(relation, candidate_tuples, candidate_ids, answer_tuple):
    logits = tf.squeeze(tf.matmul(relation, candidate_tuples, adj_y=True),squeeze_dims=1)
    answer_logit=tf.squeeze(tf.matmul(relation, answer_tuple, adj_y=True),squeeze_dims=1)
    objective = tf.nn.softplus(tf.reduce_sum(logits,1,keep_dims=True)-2*answer_logit)
    loss = tf.reduce_mean(objective, name='predictor_loss')
    batchindex= tf.expand_dims(tf.to_int64(tf.range(tf.shape(candidate_ids)[0])),-1)
    candindex = tf.expand_dims(tf.arg_max(logits,1),-1)
    indexes   = tf.concat(1,[batchindex,candindex])
    predict = tf.gather_nd(candidate_ids, indexes, name='prediction')
    return logits, loss, predict




# Aliases
bicond_singlesupport_reader = conditional_reader_model
bicond_singlesupport_reader_with_cands = conditional_reader_model_with_cands
bilstm_singlesupport_reader_with_cands = bilstm_reader_model_with_cands
bilstm_nosupport_reader_with_cands = bilstm_nosupport_reader_model_with_cands
boe_support_cands = boe_support_cands_reader_model
boe_nosupport_cands = boe_nosupport_cands_reader_model
boe_support = boe_reader_model
boe_nosupport = boenosupport_reader_model
model_f = model_f_reader_model

# Available models
__models__ = [
    'bicond_singlesupport_reader',
    'bicond_singlesupport_reader_with_cands',
    'bilstm_singlesupport_reader_with_cands',
    'bilstm_nosupport_reader_with_cands',
    'boe_multisupport_avg_reader_with_cands',
    'boe_multisupport_lossavg_reader_with_cands',
    'boe_support_cands',
    'boe_nosupport_cands',
    'boe_support',
    'boe_nosupport',
    'model_f'
]


def get_function(function_name):
    this_module = sys.modules[__name__]
    if not hasattr(this_module, function_name):
        raise ValueError('Unknown model: {}'.format(function_name))
    return getattr(this_module, function_name)
