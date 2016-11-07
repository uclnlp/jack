import tensorflow as tf
import numpy as np
import quebap.util.tfutil as tfutil
#from quebap.tensorizer import Feature_Tensorizer, MultipleChoiceReader, \
#    AtomicTensorizer, SequenceTensorizer
from quebap.projects.clozecompose.tensorizer import *
from quebap.util import tfutil as tfutil


def create_dense_embedding(ids, repr_dim, num_symbols):
    """
    :param ids: tensor [d1, ... ,dn] of int32 symbols
    :param repr_dim: dimension of embeddings
    :param num_symbols: number of symbols
    :return: [d1, ... ,dn,repr_dim] tensor representation of symbols.
    """
    embeddings = tf.Variable(tf.random_normal((num_symbols, repr_dim), dtype=tf.float64), dtype=tf.float64)
    encodings = tf.gather(embeddings, ids)  # [batch_size, repr_dim]
    return encodings


def create_sequence_embedding(inputs, seq_lengths, repr_dim, vocab_size, emb_name, rnn_scope, reuse_scope=False):
    """
    :param inputs: tensor [d1, ... ,dn] of int32 symbols
    :param seq_lengths: [s1, ..., sn] lengths of instances in the batch
    :param repr_dim: dimension of embeddings
    :param vocab_size: number of symbols
    :return: return [batch_size, repr_dim] tensor representation of symbols.
    """
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, repr_dim], -0.1, 0.1, dtype=tf.float64),
                                   name=emb_name, trainable=True, dtype=tf.float64)
    # [batch_size, max_seq_length, input_size]
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)

    # dummy test to see if the embedding lookup is working
    # Reduce along dimension 1 (`n_input`) to get a single vector (row) per input example
    # embedding_aggregated = tf.reduce_sum(embedded_inputs, [1])

    with tf.variable_scope(rnn_scope) as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell = tf.nn.rnn_cell.LSTMCell(num_units=repr_dim, state_is_tuple=True)
        # returning [batch_size, max_time, cell.output_size]
        outputs, last_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float64,
            sequence_length=seq_lengths,
            inputs=embedded_inputs)

    return tfutil.get_by_index(outputs, seq_lengths)  # [batch_size x hidden_size] tensor of last outputs



def create_bowv_embedding(inputs, repr_dim, vocab_size, emb_name):
    """
    Bag of word vector encoding (dense embedding)
    :param inputs: tensor [d1, ... ,dn] of int32 symbols
    :param seq_lengths: [s1, ..., sn] lengths of instances in the batch
    :param repr_dim: dimension of embeddings
    :param vocab_size: number of symbols
    :return: [batch_size, repr_dim]
    """

    # use a shared embedding matrix for now, test if this outperforms separate matrices later
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, repr_dim], -0.1, 0.1, dtype=tf.float64),
                                   name=emb_name, trainable=True, dtype=tf.float64)
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)

    # Reduce along dimension 1 (`n_input`) to get a single vector (row) per input example
    embedding_aggregated = tf.reduce_mean(embedded_inputs, [1])

    return embedding_aggregated


def create_bi_sequence_embedding(inputs, seq_lengths, repr_dim, vocab_size, emb_name, rnn_scope, reuse_scope=False):
    """
    Bidirectional encoding
    :param inputs: tensor [d1, ... ,dn] of int32 symbols
    :param seq_lengths: [s1, ..., sn] lengths of instances in the batch
    :param repr_dim: dimension of embeddings
    :param vocab_size: number of symbols
    :return: return outputs_fw, last_state_fw, outputs_bw, last_state_bw
    """

    # use a shared embedding matrix for now, test if this outperforms separate matrices later
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, repr_dim], -0.1, 0.1, dtype=tf.float64),
                                   name=emb_name, trainable=True, dtype=tf.float64)
    # [batch_size, max_seq_length, input_size]
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)

    # dummy test to see if the embedding lookup is working
    # Reduce along dimension 1 (`n_input`) to get a single vector (row) per input example
    # embedding_aggregated = tf.reduce_sum(embedded_inputs, [1])


    ### first FW LSTM ###
    with tf.variable_scope(rnn_scope + "_FW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_fw = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        #cell_fw = tf.contrib.rnn.AttentionCellWrapper(cell_fw, 3, state_is_tuple=True) # not working
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_fw, output_keep_prob=0.9)
        # outputs shape: [batch_size, max_time, cell.output_size]
        # last_states shape: [batch_size, cell.state_size]
        outputs_fw, last_state_fw = tf.nn.dynamic_rnn(
            cell=cell_fw,
            dtype=tf.float64,
            sequence_length=seq_lengths,
            inputs=embedded_inputs)


    embedded_inputs_rev = tf.reverse(embedded_inputs, [False, True, False])  # reverse the sequence

    ### first BW LSTM ###
    with tf.variable_scope(rnn_scope + "_BW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_bw = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_bw, output_keep_prob=0.9)

        # outputs shape: [batch_size, max_time, cell.output_size]
        # last_states shape: [batch_size, cell.state_size]
        outputs_bw, last_state_bw = tf.nn.dynamic_rnn(
            cell=cell_bw,
            dtype=tf.float64,
            sequence_length=seq_lengths,
            inputs=embedded_inputs_rev)


    return outputs_fw, last_state_fw, outputs_bw, last_state_bw, embedding_matrix




def create_bi_sequence_embedding_initialise(inputs_cond, seq_lengths_cond, repr_dim, rnn_scope_cond, last_state_fw, last_state_bw, embedding_matrix, reuse_scope=False):
    """
    Bidirectional conditional encoding
    :param inputs: tensor [d1, ... ,dn] of int32 symbols
    :param seq_lengths: [s1, ..., sn] lengths of instances in the batch
    :param repr_dim: dimension of embeddings
    :param vocab_size: number of symbols
    :return: return [batch_size, repr_dim] tensor representation of symbols.
    """

    ### second FW LSTM ###

    embedded_inputs_cond = tf.nn.embedding_lookup(embedding_matrix, inputs_cond) # [batch_size, max_seq_length, input_size]

    # initialise with state of context
    with tf.variable_scope(rnn_scope_cond + "_FW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_fw_cond = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        cell_fw_cond = tf.nn.rnn_cell.DropoutWrapper(cell=cell_fw_cond, output_keep_prob=0.9)

        # returning [batch_size, max_time, cell.output_size]
        outputs_fw_cond, last_state_fw_cond = tf.nn.dynamic_rnn(
            cell=cell_fw_cond,
            dtype=tf.float64,
            sequence_length=seq_lengths_cond,
            inputs=embedded_inputs_cond,
            initial_state=last_state_fw
        )

    embedded_inputs_cond_rev = tf.reverse(embedded_inputs_cond, [False, True, False])  # reverse the sequence


    ### second BW LSTM ###

    with tf.variable_scope(rnn_scope_cond + "_BW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_fw = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_fw, output_keep_prob=0.9)

        # outputs shape: [batch_size, max_time, cell.output_size]
        # last_states shape: [batch_size, cell.state_size]
        outputs_bw_cond, last_state_bw_cond = tf.nn.dynamic_rnn(
            cell=cell_fw,
            dtype=tf.float64,
            sequence_length=seq_lengths_cond,
            inputs=embedded_inputs_cond_rev,
            initial_state=last_state_bw
        )


    last_output_fw = tfutil.get_by_index(outputs_fw_cond, seq_lengths_cond)
    last_output_bw = tfutil.get_by_index(outputs_bw_cond, seq_lengths_cond)

    outputs_fin = tf.concat(1, [last_output_fw, last_output_bw])


    return outputs_fin





def create_bicond_sequence_embedding(inputs, seq_lengths, inputs_cond, seq_lengths_cond, repr_dim, vocab_size, emb_name, rnn_scope, rnn_scope_cond, reuse_scope=False):
    """
    Bidirectional conditional encoding
    :param inputs: tensor [d1, ... ,dn] of int32 symbols
    :param seq_lengths: [s1, ..., sn] lengths of instances in the batch
    :param repr_dim: dimension of embeddings
    :param vocab_size: number of symbols
    :return: return [batch_size, repr_dim] tensor representation of symbols.
    """

    # use a shared embedding matrix for now, test if this outperforms separate matrices later
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, repr_dim], -0.1, 0.1, dtype=tf.float64),
                                   name=emb_name, trainable=True, dtype=tf.float64)
    # [batch_size, max_seq_length, input_size]
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)

    # dummy test to see if the embedding lookup is working
    # Reduce along dimension 1 (`n_input`) to get a single vector (row) per input example
    # embedding_aggregated = tf.reduce_sum(embedded_inputs, [1])


    ### first FW LSTM ###
    with tf.variable_scope(rnn_scope + "_FW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_fw = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        # outputs shape: [batch_size, max_time, cell.output_size]
        # last_states shape: [batch_size, cell.state_size]
        outputs_fw, last_state_fw = tf.nn.dynamic_rnn(
            cell=cell_fw,
            dtype=tf.float64,
            sequence_length=seq_lengths,
            inputs=embedded_inputs)


    ### second FW LSTM ###

    embedded_inputs_cond = tf.nn.embedding_lookup(embedding_matrix, inputs_cond) # [batch_size, max_seq_length, input_size]

    # initialise with state of context
    with tf.variable_scope(rnn_scope_cond + "_FW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_fw_cond = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        # returning [batch_size, max_time, cell.output_size]
        outputs_fw_cond, last_state_fw_cond = tf.nn.dynamic_rnn(
            cell=cell_fw_cond,
            dtype=tf.float64,
            sequence_length=seq_lengths_cond,
            inputs=embedded_inputs_cond,
            initial_state=last_state_fw
        )

    embedded_inputs_rev = tf.reverse(embedded_inputs, [False, True, False])  # reverse the sequence

    ### first BW LSTM ###
    with tf.variable_scope(rnn_scope + "_BW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_fw = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        # outputs shape: [batch_size, max_time, cell.output_size]
        # last_states shape: [batch_size, cell.state_size]
        outputs_bw, last_state_bw = tf.nn.dynamic_rnn(
            cell=cell_fw,
            dtype=tf.float64,
            sequence_length=seq_lengths,
            inputs=embedded_inputs_rev)


    embedded_inputs_cond_rev = tf.reverse(embedded_inputs_cond, [False, True, False])  # reverse the sequence

    ### second BW LSTM ###
    with tf.variable_scope(rnn_scope_cond + "_BW") as scope:
        if reuse_scope == True:
            scope.reuse_variables()
        cell_fw = tf.nn.rnn_cell.LSTMCell(repr_dim, state_is_tuple=True)
        # outputs shape: [batch_size, max_time, cell.output_size]
        # last_states shape: [batch_size, cell.state_size]
        outputs_bw_cond, last_state_bw_cond = tf.nn.dynamic_rnn(
            cell=cell_fw,
            dtype=tf.float64,
            sequence_length=seq_lengths_cond,
            inputs=embedded_inputs_cond_rev,
            initial_state=last_state_bw
        )


    last_output_fw = tfutil.get_by_index(outputs_fw_cond, seq_lengths_cond)
    last_output_bw = tfutil.get_by_index(outputs_bw_cond, seq_lengths_cond)

    outputs_fin = tf.concat(1, [last_output_fw, last_output_bw])

    return outputs_fin



def create_dot_product_scorer(question_encodings, candidate_encodings):
    """

    :param question_encodings: [batch_size, enc_dim] tensor of question representations
    :param candidate_encodings: [batch_size, num_candidates, enc_dim] tensor of candidate encodings
    :return: a [batch_size, num_candidate] tensor of scores for each candidate
    """
    return tf.reduce_sum(tf.mul(tf.expand_dims(question_encodings, 1), candidate_encodings), 2)  # tf.mul same as * except it does backprop correctly


def create_softmax_loss(scores, target_values):
    """

    :param scores: [batch_size, num_candidates] logit scores
    :param target_values: [batch_size, num_candidates] vector of 0/1 target values.
    :return: [batch_size] vector of losses (or single number of total loss).
    """
    return tf.nn.softmax_cross_entropy_with_logits(scores, target_values)




def create_bag_of_embeddings_reader(reference_data, **options):
    """
    A reader that creates sequence representations of the input reading instance, and then
    models each question and candidate as the sum of the embeddings of their tokens.
    :param reference_data: the reference training set that determines the vocabulary.
    :param options: repr_dim, candidate_split (used for tokenizing candidates), question_split
    :return: a MultipleChoiceReader.
    """
    tensorizer = SequenceTensorizer(reference_data)

    # get embeddings for each question token
    # [batch_size, max_question_length, repr_dim]
    question_embeddings = create_dense_embedding(tensorizer.questions, options['repr_dim'], tensorizer.num_symbols)
    question_encoding = tf.reduce_sum(question_embeddings, 1)  # [batch_size, repr_dim]

    # [batch_size, num_candidates, max_question_length, repr_dim
    candidate_embeddings = create_dense_embedding(tensorizer.candidates, options['repr_dim'],
                                                  tensorizer.num_symbols)
    candidate_encoding = tf.reduce_sum(candidate_embeddings, 2)  # [batch_size, num_candidates, repr_dim]
    scores = create_dot_product_scorer(question_encoding, candidate_encoding)
    loss = create_softmax_loss(scores, tensorizer.target_values)
    return MultipleChoiceReader(tensorizer, scores, loss)


def create_sequence_embeddings_reader(reference_data, **options):
    """
    A reader that creates sequence representations of the input reading instance, and then
    models each question as a sequence encoded with an RNN and candidate as the sum of the embeddings of their tokens.
    :param reference_data: the reference training set that determines the vocabulary.
    :param options: repr_dim, candidate_split (used for tokenizing candidates), question_split
    :return: a MultipleChoiceReader.
    """
    #TODO: create separate methods for model variants
    #tensorizer = SequenceTensorizer(reference_data)
    tensorizer = SequenceTensorizer(reference_data)

    dim1ql, dim2ql = tf.unpack(tf.shape(tensorizer.question_lengths))
    question_lengths_true = tf.reshape(tf.slice(tensorizer.question_lengths, [0, 0], [dim1ql, 1]), [-1])

    dim1q, dim2q, dim3q = tf.unpack(tf.shape(tensorizer.questions))
    questions_true = tf.reshape(tf.slice(tensorizer.questions, [0, 0, 0], [dim1q, 1, dim3q]), [dim1q, -1])

    dim1t, dim2t, dim3t = tf.unpack(tf.shape(tensorizer.target_values))
    targets_true = tf.reshape(tf.slice(tensorizer.target_values, [0, 0, 0], [dim1t, 1, dim3t]), [dim1t, -1])

    question_lengths_false = tf.reshape(tf.slice(tensorizer.question_lengths, [0, 1], [dim1ql, 1]), [-1])
    questions_false = tf.reshape(tf.slice(tensorizer.questions, [0, 1, 0], [dim1q, 1, dim3q]), [dim1q, -1])
    targets_false = tf.reshape(tf.slice(tensorizer.target_values, [0, 1, 0], [dim1t, 1, dim3t]), [dim1t, -1])

    # 1) bidirectional conditional encoding with one support
    #question_encoding_true = create_bicond_question_encoding(tensorizer, questions_true, question_lengths_true, options, reuse_scope=False)
    #question_encoding_false = create_bicond_question_encoding(tensorizer, questions_false, question_lengths_false, options, reuse_scope=True)
    #cand_dim = options['repr_dim']*2

    # 2) question only lstm encoding
    # true and false use same parameters
    #question_encoding_true = create_sequence_embedding(questions_true, question_lengths_true, options['repr_dim'],
    #                                              tensorizer.num_symbols, "embedding_matrix_q", "RNN_q", reuse_scope=False)
    #question_encoding_false = create_sequence_embedding(questions_false, question_lengths_false, options['repr_dim'],
    #                                                   tensorizer.num_symbols, "embedding_matrix_q", "RNN_q", reuse_scope=True)
    #cand_dim = options['repr_dim']

    # 3) all candidate unidirectional lstm encoding
    #dim1s, dim2s, dim3s = tf.unpack(tf.shape(tensorizer.support))  # [batch_size, num_supports, num_tokens]
    #sup = tf.reshape(tensorizer.support, [-1, dim3s])  # [batch_size * num_supports, num_tokens]
    #sup_l = tf.reshape(tensorizer.support_lengths, [-1])   # [support_lengths * num_supports]

    #sup_encoding = create_sequence_embedding(sup, sup_l, options['repr_dim'], tensorizer.num_symbols, "embedding_matrix_q", "RNN_q", reuse_scope=False)
    #cand_dim = options['repr_dim']

    #sup_encoding_reshaped = tf.reshape(sup_encoding, [dim1s, dim2s, cand_dim])  # [batch_size, num_supports, output_dim]
    #question_encoding = tf.reduce_mean(sup_encoding_reshaped, 1) # [batch_size, output_dim]  <-- support sequence encodings are mean averaged

    # 4) bidirectional conditional encoding with all supports averaged
    question_encoding_true = get_bicond_multisupport_question_encoding(tensorizer, questions_true, question_lengths_true, options, reuse_scope=False)
    question_encoding_false = get_bicond_multisupport_question_encoding(tensorizer, questions_false, question_lengths_false, options, reuse_scope=True)
    cand_dim = options['repr_dim'] * 2

    # [batch_size, num_candidates, max_question_length, repr_dim
    candidate_embeddings = create_dense_embedding(tensorizer.candidates, cand_dim,
                                                  tensorizer.num_symbols)
    candidate_encoding = tf.reduce_sum(candidate_embeddings, 2)  # [batch_size, num_candidates, repr_dim]

    scores_true = create_dot_product_scorer(question_encoding_true, candidate_encoding)  # a [batch_size, num_candidate] tensor of scores for each candidate
    scores_false = create_dot_product_scorer(question_encoding_false, candidate_encoding)

    loss_true = create_softmax_loss(scores_true, targets_true)
    loss_false = create_softmax_loss(scores_false, targets_false)
    loss = loss_true + loss_false

    return MultipleChoiceReader(tensorizer, scores_true, loss)




def create_bowv_embeddings_reader(reference_data, **options):
    """
    A reader that creates bowv representations of the input reading instance, and
    models each question, context and candidate as the sum of the embeddings of their tokens.
    :param reference_data: the reference training set that determines the vocabulary.
    :param options: repr_dim, candidate_split (used for tokenizing candidates), question_split
    :return: a MultipleChoiceReader.
    """
    tensorizer = SequenceTensorizer(reference_data)

    dim1ql, dim2ql = tf.unpack(tf.shape(tensorizer.question_lengths))
    #question_lengths_true = tf.squeeze(tf.slice(tensorizer.question_lengths, [0, 0], [dim1ql, 1]), [1])

    dim1q, dim2q, dim3q = tf.unpack(tf.shape(tensorizer.questions))
    questions_true = tf.squeeze(tf.slice(tensorizer.questions, [0, 0, 0], [dim1q, 1, dim3q]), [1])

    dim1t, dim2t, dim3t = tf.unpack(tf.shape(tensorizer.target_values))
    targets_true = tf.squeeze(tf.slice(tensorizer.target_values, [0, 0, 0], [dim1t, 1, dim3t]), [1])

    #question_lengths_false = tf.squeeze(tf.slice(tensorizer.question_lengths, [0, 1], [dim1ql, 1]), [1])
    questions_false = tf.squeeze(tf.slice(tensorizer.questions, [0, 1, 0], [dim1q, 1, dim3q]), [1])
    targets_false = tf.squeeze(tf.slice(tensorizer.target_values, [0, 1, 0], [dim1t, 1, dim3t]), [1])

    # 5) bag of word vector encoding with all supports averaged
    question_encoding_true = get_bowv_multisupport_question_encoding(tensorizer, questions_true, options)
    question_encoding_false = get_bowv_multisupport_question_encoding(tensorizer, questions_false, options)
    cand_dim = options['repr_dim']

    # [batch_size, num_candidates, max_question_length, repr_dim
    candidate_embeddings = create_dense_embedding(tensorizer.candidates, cand_dim,
                                                  tensorizer.num_symbols)
    candidate_encoding = tf.reduce_sum(candidate_embeddings, 2)  # [batch_size, num_candidates, repr_dim]

    scores_true = create_dot_product_scorer(question_encoding_true, candidate_encoding)  # a [batch_size, num_candidate] tensor of scores for each candidate
    scores_false = create_dot_product_scorer(question_encoding_false, candidate_encoding)

    loss_true = create_softmax_loss(scores_true, targets_true)
    loss_false = create_softmax_loss(scores_false, targets_false)

    # add scores and losses for pos and neg examples
    #scores_all = scores_true + scores_false
    loss_all = loss_true + loss_false

    loss_all = loss_all + tf.scalar_mul(0.1, tf.nn.l2_loss(loss_all))

    return MultipleChoiceReader(tensorizer, scores_true, loss_all)



def create_bowv_nosupport_embeddings_reader(reference_data, **options):
    """
    A reader that creates bowv representations of the input reading instance, and
    models each question and candidate as the sum of the embeddings of their tokens. Support is ignored.
    :param reference_data: the reference training set that determines the vocabulary.
    :param options: repr_dim, candidate_split (used for tokenizing candidates), question_split
    :return: a MultipleChoiceReader.
    """
    tensorizer = SequenceTensorizer(reference_data)

    dim1ql, dim2ql = tf.unpack(tf.shape(tensorizer.question_lengths))
    #question_lengths_true = tf.squeeze(tf.slice(tensorizer.question_lengths, [0, 0], [dim1ql, 1]), [1])

    dim1q, dim2q, dim3q = tf.unpack(tf.shape(tensorizer.questions))
    questions_true = tf.squeeze(tf.slice(tensorizer.questions, [0, 0, 0], [dim1q, 1, dim3q]), [1])

    dim1t, dim2t, dim3t = tf.unpack(tf.shape(tensorizer.target_values))
    targets_true = tf.squeeze(tf.slice(tensorizer.target_values, [0, 0, 0], [dim1t, 1, dim3t]), [1])

    #question_lengths_false = tf.squeeze(tf.slice(tensorizer.question_lengths, [0, 1], [dim1ql, 1]), [1])
    questions_false = tf.squeeze(tf.slice(tensorizer.questions, [0, 1, 0], [dim1q, 1, dim3q]), [1])
    targets_false = tf.squeeze(tf.slice(tensorizer.target_values, [0, 1, 0], [dim1t, 1, dim3t]), [1])

    # 5) bag of word vector encoding with all supports averaged
    #question_encoding_true = get_bowv_multisupport_question_encoding(tensorizer, questions_true, options)
    #question_encoding_false = get_bowv_multisupport_question_encoding(tensorizer, questions_false, options)

    cand_dim = options['repr_dim']
    #question_encoding_true = create_dense_embedding(questions_false, cand_dim, tensorizer.num_symbols)
    #question_encoding_true = tf.reduce_sum(question_encoding_true, 2)

    #question_encoding_false = create_dense_embedding(questions_true, cand_dim, tensorizer.num_symbols)
    #question_encoding_false = tf.reduce_sum(question_encoding_false, 2)
    question_encoding_true = get_bowv_question_encoding(tensorizer, questions_true, options)
    question_encoding_false = get_bowv_question_encoding(tensorizer, questions_false, options)

    # [batch_size, num_candidates, max_question_length, repr_dim
    candidate_embeddings = create_dense_embedding(tensorizer.candidates, cand_dim,
                                                  tensorizer.num_symbols)
    candidate_encoding = tf.reduce_sum(candidate_embeddings, 2)  # [batch_size, num_candidates, repr_dim]

    scores_true = create_dot_product_scorer(question_encoding_true, candidate_encoding)  # a [batch_size, num_candidate] tensor of scores for each candidate
    scores_false = create_dot_product_scorer(question_encoding_false, candidate_encoding)

    loss_true = create_softmax_loss(scores_true, targets_true)
    loss_false = create_softmax_loss(scores_false, targets_false)

    # add scores and losses for pos and neg examples
    loss = loss_true + loss_false

    return MultipleChoiceReader(tensorizer, scores_true, loss)



def create_bicond_question_encoding(tensorizer, questions_true, question_lengths_true, options, reuse_scope=False):
    dim1s, dim2s, dim3s = tf.unpack(tf.shape(tensorizer.support))  # [batch_size, num_supports, num_tokens]

    #sup = tf.squeeze(tf.slice(tensorizer.support, [0, 0, 0], [dim1s, 1, dim3s]), [1])  # take the first support, this is the middle dimension
    #sup_l = tf.squeeze(tf.slice(tensorizer.support_lengths, [0, 0], [dim1s, 1]), [1])

    # Is this more efficient?
    sup = tf.reshape(tf.slice(tensorizer.support, [0, 0, 0], [dim1s, 1, dim3s]), [dim1s, -1]) # take the first support, this is the middle dimension
    sup_l = tf.reshape(tf.slice(tensorizer.support_lengths, [0, 0], [dim1s, 1]), [-1])


    # this is a question_context encoding, i.e. context conditioned on question
    # inputs, seq_lengths, inputs_cond, seq_lengths_cond, repr_dim, vocab_size, emb_name, rnn_scope, rnn_scope_cond
    question_encoding = create_bicond_sequence_embedding(questions_true, question_lengths_true, sup,
                                                         sup_l, options['repr_dim'],
                                                         tensorizer.num_symbols,
                                                         "embedding_matrix_q_c", "RNN_q", "RNN_c", reuse_scope)

    return question_encoding


def get_bowv_question_encoding(tensorizer, questions, options):
    # question only encoder

    cand_dim = options['repr_dim']

    # 4) run question encoder
    outputs_que = create_bowv_embedding(questions, cand_dim, tensorizer.num_symbols, "embedding_matrix_que")

    # 5) combine and return
    #repr = tf.reshape((outputs_que), [-1, cand_dim])

    return outputs_que


def get_bowv_multisupport_question_encoding(tensorizer, questions, options):
    # bowv encoder
    cand_dim = options['repr_dim']

    # 1) reshape the support tensors to remove batch_size dimension
    dim1s, dim2s, dim3s = tf.unpack(tf.shape(tensorizer.support))  # [batch_size, num_supports, num_tokens]
    sup = tf.reshape(tensorizer.support, [dim1s * dim2s, dim3s])  # [batch_size * num_supports, num_tokens]

    # 2) run first rnn to encode the supports
    outputs_sup = create_bowv_embedding(sup, cand_dim, tensorizer.num_symbols, "embedding_matrix_sup")

    # 3) reshape the outputs of the support encoder to average outputs by batch_size
    outputs_sup = tf.reshape(outputs_sup, [dim1s, dim2s, cand_dim])  # [batch_size, num_supports, cell.state_size]
    outputs_sup = tf.reduce_mean(outputs_sup, 1)  # [batch_size, cell.state_size]  <-- support encodings are mean averaged

    # 4) run question encoder
    outputs_que = create_bowv_embedding(questions, cand_dim, tensorizer.num_symbols, "embedding_matrix_que")

    # 5) combine and return
    repr = tf.reshape(outputs_que * outputs_sup, [dim1s, cand_dim])

    return repr



def get_bicond_multisupport_question_encoding(tensorizer, questions_true, question_lengths_true, options, reuse_scope=False):
    # bidirectional conditional encoding with all supports averaged

    cand_dim = options['repr_dim']

    # 1) reshape the support tensors to remove batch_size dimension

    dim1s, dim2s, dim3s = tf.unpack(tf.shape(tensorizer.support))  # [batch_size, num_supports, num_tokens]
    sup = tf.reshape(tensorizer.support, [-1, dim3s])  # [batch_size * num_supports, num_tokens]
    sup_l = tf.reshape(tensorizer.support_lengths, [-1])  # [support_lengths * num_supports]


    # 2) run first rnn to encode the supports
    outputs_fw, last_state_fw, outputs_bw, last_state_bw, embedding_matrix = create_bi_sequence_embedding(sup, sup_l, cand_dim,
                                                         tensorizer.num_symbols,
                                                         "embedding_matrix_q_c", "RNN_c", reuse_scope)


    # 3) reshape the outputs of the bi-lstm support encoder to average outputs by batch_size

    (last_state_fw_c, last_state_fw_h) = last_state_fw
    (last_state_bw_c, last_state_bw_h) = last_state_bw

    last_state_fw_c_reshaped = tf.reshape(last_state_fw_c, [dim1s, dim2s, cand_dim])  # [batch_size, num_supports, cell.state_size]
    last_state_fw_c = tf.reduce_mean(last_state_fw_c_reshaped, 1)  # [batch_size, cell.state_size]  <-- support state encodings are mean averaged
    last_state_fw_h_reshaped = tf.reshape(last_state_fw_h, [dim1s, dim2s, cand_dim])  # [batch_size, num_supports, cell.state_size]
    last_state_fw_h = tf.reduce_mean(last_state_fw_h_reshaped, 1)  # [batch_size, cell.state_size]  <-- support state encodings are mean averaged
    sup_states_fw = tf.nn.rnn_cell.LSTMStateTuple(last_state_fw_c, last_state_fw_h)
    # normal Python tuples used to work, but now that throws "First structure has type type 'tuple', while second structure has type class 'tensorflow.python.ops.rnn_cell.LSTMStateTuple'"

    last_state_bw_c_reshaped = tf.reshape(last_state_bw_c, [dim1s, dim2s, cand_dim])  # [batch_size, num_supports, cell.state_size]
    last_state_bw_c = tf.reduce_mean(last_state_bw_c_reshaped, 1)  # [batch_size, cell.state_size]  <-- support state encodings are mean averaged
    last_state_bw_h_reshaped = tf.reshape(last_state_bw_h, [dim1s, dim2s, cand_dim])  # [batch_size, num_supports, cell.state_size]
    last_state_bw_h = tf.reduce_mean(last_state_bw_h_reshaped, 1)  # [batch_size, cell.state_size]  <-- support state encodings are mean averaged
    sup_states_bw = tf.nn.rnn_cell.LSTMStateTuple(last_state_bw_c, last_state_bw_h)


    # 4) feed into bi-lstm question encoder

    support_question_encoding = create_bi_sequence_embedding_initialise(questions_true, question_lengths_true, cand_dim, "RNN_q", sup_states_fw, sup_states_bw, embedding_matrix, reuse_scope)


    # 5) profit!

    return support_question_encoding




def create_support_bag_of_embeddings_reader(reference_data, **options):
    """
    A reader that creates sequence representations of the input reading instance, and then
    models each question and candidate as the sum of the embeddings of their tokens.
    :param reference_data: the reference training set that determines the vocabulary.
    :param options: repr_dim, candidate_split (used for tokenizing candidates), question_split
    :return: a MultipleChoiceReader.
    """
    tensorizer = SequenceTensorizer(reference_data)

    candidate_dim = options['repr_dim']
    support_dim = options['support_dim']

    # question embeddings: for each symbol a [support_dim, candidate_dim] matrix
    question_embeddings = tf.Variable(tf.random_normal((tensorizer.num_symbols, support_dim, candidate_dim), dtype=tf.float64), dtype=tf.float64)

    # [batch_size, max_question_length, support_dim, candidate_dim]
    question_encoding_raw = tf.gather(question_embeddings, tensorizer.questions)

    # question encoding should have shape: [batch_size, 1, support_dim, candidate_dim], so reduce and keep
    question_encoding = tf.reduce_sum(question_encoding_raw, 1, keep_dims=True)

    # candidate embeddings: for each symbol a [candidate_dim] vector
    candidate_embeddings = tf.Variable(tf.random_normal((tensorizer.num_symbols, candidate_dim), dtype=tf.float64), dtype=tf.float64)
    # [batch_size, num_candidates, max_candidate_length, candidate_dim]
    candidate_encoding_raw = tf.gather(candidate_embeddings, tensorizer.candidates)

    # candidate embeddings should have shape: [batch_size, num_candidates, 1, candidate_dim]
    candidate_encoding = tf.reduce_sum(candidate_encoding_raw, 2, keep_dims=True)

    # each symbol has [support_dim] vector
    support_embeddings = tf.Variable(tf.random_normal((tensorizer.num_symbols, support_dim), dtype=tf.float64), dtype=tf.float64)

    # [batch_size, max_support_num, max_support_length, support_dim]
    support_encoding_raw = tf.gather(support_embeddings, tensorizer.support)

    # support encoding should have shape: [batch_size, 1, support_dim, 1]
    support_encoding = tf.expand_dims(tf.expand_dims(tf.reduce_sum(support_encoding_raw, (1, 2)), 1), 3)

    # scoring with a dot product
    # [batch_size, num_candidates, support_dim, candidate_dim]
    combined = question_encoding * candidate_encoding * support_encoding
    scores = tf.reduce_sum(combined, (2, 3))

    loss = create_softmax_loss(scores, tensorizer.target_values)
    return MultipleChoiceReader(tensorizer, scores, loss)





class SequenceTensorizerTokens2(Tensorizer):
    """
    Converts reading instances into tensors of integer sequences representing tokens. A question batch
    is tranformed into a [batch_size, max_length] integer matrix (question placeholder),
    a list of candidates into a [batch_size, num_candidates, max_length] integer tensor (candidates placeholder)
    the answers are a 0/1 float [batch_size, num_candidates] matrix indicating a true (1) or false (0) label
    for each candidate. (target_values placeholder)
    The difference with respect to the SequenceTensorizer is that question lengths are included, for use with the
    Tensorflow dynamic_rnn
    """

    def __init__(self, reference_data):
        """
        Create a new SequenceTensorizer.
        :param reference_data: the training data that determines the lexicon.
        :param candidate_split: the regular expression used for tokenizing candidates.
        :param question_split: the regular expression used for tokenizing questions.
        :param support_split: the regular expression used for tokenizing support documents.
        """
        self.useSupport = True
        self.reference_data = reference_data
        self.pad = "<pad>"
        self.none = "<none>"  # for NONE answer / neg instances

        self.question_lengths = tf.placeholder(tf.int32, (None, None), name="question_lengths")  # [batch_size, pos/neg]
        #self.candidate_lengths = tf.placeholder(tf.int32, (None, None), name="candidate_lengths")  # [batch_size, num_candidates]
        self.support_lengths = tf.placeholder(tf.int32, (None, None), name="support_lengths")  # [batch_size, num_support]
        self.support = tf.placeholder(tf.int32, (None, None, None), name="support")  # [batch_size, num_supports, num_tokens]


        self.questions = tf.placeholder(tf.int32, (None, None, None), name="question")  # [batch_size, pos/neg, num_tokens]
        #self.candidates = tf.placeholder(tf.int32, (None, None), name="candidates")  # [batch_size, num_candidates]
        self.target_values = tf.placeholder(tf.float64, (None, None, None), name="target") # [batch_size, num_tokens, num_types]

        self.target_lengths = tf.placeholder(tf.int64, (None), name="target_lengths")  # [batch_size]

        instances = reference_data['instances']

        all_question_tokens = [self.pad, self.none] + [token
                                                        for inst in instances
                                                        for question in inst['questions']
                                                        for token in
                                                        word_tokenize(question['question'])]

        all_support_tokens = [self.pad, self.none] + [token
                                                       for inst in instances
                                                       for support in inst['support']
                                                       for token in
                                                       word_tokenize(support['text'])]

        """all_candidate_tokens = [self.pad, self.none] + [token
                                                         for inst in instances
                                                         for question in inst['questions']
                                                         for candidate in question['candidates'] + question['answers']
                                                         for token in
                                                         word_tokenize(candidate['text'])]"""


        count = [[self.pad, -1], [self.none, -1]]
        count.extend(collections.Counter(all_question_tokens + all_support_tokens).most_common(50000-2))  # 50000

        self.all_tokens = [t[0] for t in count]

        self.lexicon = FrozenIdentifier(self.all_tokens, default_key=self.none)
        self.num_symbols = len(self.lexicon)


        all_question_seqs = [[self.lexicon[t]
                              for t in word_tokenize(inst['questions'][0]['question'])]
                             for inst in instances]

        self.all_max_question_length = max([len(q) for q in all_question_seqs])

        self.all_question_seqs_padded = [pad_seq(q, self.all_max_question_length, self.lexicon[self.pad]) for q in all_question_seqs]

        self.random = random.Random(0)


    def create_batches(self, data=None, batch_size=1, test=False):
        """
        Take a dataset and create a generator of (batched) feed_dict objects. At training time this
        tensorizer sub-samples the candidates (currently one positive and one negative candidate).
        :param data: the input dataset.
        :param batch_size: size of each batch.
        :param test: should this be generated for test time? If so, the candidates are all possible candidates.
        :return: a generator of batches.
        """

        instances = self.reference_data['instances'] if data is None else data['instances']

        for b in range(0, len(instances) // batch_size):
            batch = instances[b * batch_size: (b + 1) * batch_size]


            support_seqs = [[[self.lexicon[t]
                          for t in word_tokenize(support['text'])]
                         for support in inst['support']]
                        for inst in batch]
            max_support_length = max([len(a) for support in support_seqs for a in support])
            max_num_support = max([len(supports) for supports in support_seqs])
            # [batch_size, max_num_support, max_support_length]
            empty_support = pad_seq([], max_support_length, self.lexicon[self.pad])
            # support_seqs_padded = [self.pad_seq([self.pad_seq(s, max_support_length) for s in supports], max_num_support) for supports in support_seqs]
            support_seqs_padded = [
                pad_seq([pad_seq(s, max_support_length, self.lexicon[self.pad]) for s in batch_item],
                        max_num_support, empty_support)
                for batch_item in support_seqs]

            support_length = [[len(c) for c in pad_seq(inst, max_num_support, [])] for inst in support_seqs]

            answer_seqs = [[[self.lexicon[t]
                              for t in word_tokenize(answ['text'])]
                             for answ in inst['questions'][0]['answers']]
                            for inst in batch]

            question_seqs = [[self.lexicon[t]
                           for t in word_tokenize(inst['questions'][0]['question'])]
                         for inst in batch]

            #candidate_seqs = [[[self.lexicon[t]
            #                    for t in word_tokenize(candidate['text'])]
            #                   for candidate in inst['questions'][0]['candidates']]
            #                  for inst in batch]

            #max_question_length = max([len(q) for q in question_seqs])
            max_question_length = self.all_max_question_length
            #max_answer_length = max([len(a) for answer in answer_seqs for a in answer])



            #max_candidate_length = max([len(a) for cand in candidate_seqs for a in cand])
            #max_num_cands = max([len(cands) for cands in candidate_seqs])
            #max_num_answs = max([len(answs) for answs in answer_seqs])

            # [batch_size, max_question_length]
            question_seqs_padded = [pad_seq(q, max_question_length, self.lexicon[self.pad]) for q in question_seqs]

            neg_question_seqs_padded = []
            for qi in question_seqs_padded:
                rq = []
                while rq == [] or rq == qi:
                    rq = self.random.choice(self.all_question_seqs_padded)
                neg_question_seqs_padded.append(rq)

            # target is a candidate-length vector of 0/1s
            # target_values_padded = [[c for c in pad_seq(inst, max_num_cands, 0.0)] for inst in target_values]

            question_length = [len(q) for q in question_seqs]
            # todo: change to actual lengths
            question_length_neg = [len(q) for q in neg_question_seqs_padded]

            # [batch_size, max_num_cands, max_candidate_length]
            #empty_candidates = pad_seq([], max_candidate_length, self.lexicon[self.pad])
            #candidate_seqs_padded = [
            #    pad_seq([pad_seq(s, max_candidate_length, self.lexicon[self.pad]) for s in batch_item], max_num_cands, empty_candidates)
            #    for batch_item in candidate_seqs]


            # A [batch_size, targ_len] float matrix representing the I/O state of each token using 1 / 0 values
            # rewrite to work with list comprehension

            #should we expand this maybe?
            target_values = [[transSentToIO(word_tokenize(support['text']), inst['questions'][0]['answers'])
                              for support in inst['support']]
                             for inst in batch]

            empty_targets = pad_seq([], max_support_length, [0.0, 0.0])
            target_values_padded = [
                pad_seq([pad_seq(s, max_support_length, [0.0, 0.0]) for s in batch_item],
                        max_num_support, empty_targets)
                for batch_item in target_values]

            target_values = [list(chain.from_iterable(subl)) for subl in target_values_padded]
            target_len = [len(t) for t in target_values]



            target_values_padded = [pad_seq(t, max(target_len), 0.0) for t in target_values]

            """
            target_values = [[transSentToIO(word_tokenize(support['text']), inst['questions'][0]['answers'])
                              for support in inst['support']]
                             for inst in batch]

            target_values_padded = [pad_seq([pad_seq(s, max_support_length, 0.0) for s in batch_item],
                                        max_num_support, 0.0)
                                for batch_item in target_values]"""


            # to test dimensionalities
            """print(tf.shape(self.questions), tf.shape(question_seqs_padded))
            print(tf.shape(self.question_lengths), tf.shape(question_length))
            print(tf.shape(self.candidates), tf.shape(candidate_seqs_padded))
            print(tf.shape(self.candidate_lengths), tf.shape(candidate_length))
            print(tf.shape(self.support), tf.shape(support_seqs_padded))
            print(tf.shape(self.target_values), tf.shape(target_values_padded))"""


            # target values for test are not supplied, performance at test time is estimated by printing to converting back to quebaps again

            if test:
                yield {
                self.questions: question_seqs_padded,
                self.question_lengths: question_length,
                #self.candidates: candidate_seqs_padded,  # !!! also fix in main code
                #self.candidate_lengths: candidate_length,
                self.support: support_seqs_padded,
                self.support_lengths: support_length,
                self.target_lengths: target_len
                }
            else:
                yield {
                self.questions: [(pos, neg) for pos, neg in zip(question_seqs_padded, neg_question_seqs_padded)],
                self.question_lengths: [(pos, neg) for pos, neg in zip(question_length, question_length_neg)],
                #self.candidates: candidate_seqs_padded,
                #self.candidate_lengths: candidate_length,
                self.target_values: target_values_padded,  #[(1.0, 0.0) for _ in range(0, batch_size)],
                self.support: support_seqs_padded,
                self.support_lengths: support_length,
                self.target_lengths: target_len
                }

    #def pad_seq(self, seq, target_length):
    #    return pad_seq(seq, target_length, self.pad)


    def convert_to_predictions(self, batch, scores):
        """
        Convert a batched candidate tensor and batched scores back into a python dictionary in quebap format.
        :param candidates: candidate representation as generated by this tensorizer.
        :param scores: scores tensor of the shape of the target_value placeholder.
        :return: sequence of reading instances corresponding to the input.
        """
        candidates = batch[self.candidates]
        all_results = []
        for scores_per_question, candidates_per_question in zip(scores, candidates):
            result_for_question = []
            for score, candidate_seq in zip(scores_per_question, candidates_per_question):
                candidate_tokens = [self.lexicon.key_by_id(sym) for sym in candidate_seq if
                                    sym != self.lexicon[self.pad]]
                candidate_text = " ".join(candidate_tokens)
                candidate = {
                    'text': candidate_text,
                    'score': score
                }
                result_for_question.append(candidate)
            question = {'answers': sorted(result_for_question, key=lambda x: -x['score'])}
            quebap = {'questions': [question]}
            all_results.append(quebap)
        return all_results






class SequenceTensorizerTokens(Tensorizer):
    """
    Converts reading instances into tensors of integer sequences representing tokens. A question batch
    is tranformed into a [batch_size, num_questions, max_length] integer matrix (question placeholder),
    there is no candidate placeholder (!),
    the answers are a [batch_size, num_questions, max_len] matrix indicating a true (1) or false (0) label
    for each token. (target_values placeholder).
    The latter is automatically constructured from candidates in the json files
    """

    def __init__(self, reference_data):
        """
        Create a new SequenceTensorizer.
        :param reference_data: the training data that determines the lexicon.
        :param candidate_split: the regular expression used for tokenizing candidates.
        :param question_split: the regular expression used for tokenizing questions.
        :param support_split: the regular expression used for tokenizing support documents.
        """
        self.useSupport = True
        self.reference_data = reference_data
        self.pad = "<pad>"
        self.none = "<none>"  # for NONE answer / neg instances

        self.question_lengths = tf.placeholder(tf.int32, (None, None), name="question_lengths")  # [batch_size, pos/neg]
        self.support_lengths = tf.placeholder(tf.int32, (None, None), name="support_lengths")  # [batch_size, num_support]
        self.support = tf.placeholder(tf.int32, (None, None, None), name="support")  # [batch_size, num_supports, num_tokens]

        self.questions = tf.placeholder(tf.int32, (None, None, None), name="question")  # [batch_size, pos/neg, num_tokens]
        self.target_values = tf.placeholder(tf.float64, (None, None, None), name="target") # [batch_size, num_support, num_tokens]


        #super().__init__(candidates, questions, target_values, support)


        instances = reference_data['instances']

        all_question_tokens = [self.pad, self.none] + [token
                                                        for inst in instances
                                                        for question in inst['questions']
                                                        for token in
                                                        word_tokenize(question['question'])]

        all_support_tokens = [self.pad, self.none] + [token
                                                       for inst in instances
                                                       for support in inst['support']
                                                       for token in
                                                       word_tokenize(support['text'])]

        count = [[self.pad, -1], [self.none, -1]]
        count.extend(collections.Counter(all_question_tokens + all_support_tokens).most_common(50000-2))  # 50000

        self.all_tokens = [t[0] for t in count]



        self.lexicon = FrozenIdentifier(self.all_tokens, default_key=self.none)
        self.num_symbols = len(self.lexicon)


        all_question_seqs = [[self.lexicon[t]
                              for t in word_tokenize(inst['questions'][0]['question'])]
                             for inst in instances]

        self.all_max_question_length = max([len(q) for q in all_question_seqs])

        self.all_question_seqs_padded = [pad_seq(q, self.all_max_question_length, self.lexicon[self.pad]) for q in all_question_seqs]

        self.random = random.Random(0)


    def create_batches(self, data=None, batch_size=1, test=False):
        """
        Take a dataset and create a generator of (batched) feed_dict objects. At training time this
        tensorizer sub-samples the candidates (currently one positive and one negative candidate).
        :param data: the input dataset.
        :param batch_size: size of each batch.
        :param test: should this be generated for test time? If so, the candidates are all possible candidates.
        :return: a generator of batches.
        """

        instances = self.reference_data['instances'] if data is None else data['instances']

        for b in range(0, len(instances) // batch_size):
            batch = instances[b * batch_size: (b + 1) * batch_size]

            support_seqs = [[[self.lexicon[t]
                          for t in word_tokenize(support['text'])]
                         for support in inst['support']]
                        for inst in batch]
            max_support_length = max([len(a) for support in support_seqs for a in support])
            max_num_support = max([len(supports) for supports in support_seqs])
            # [batch_size, max_num_support, max_support_length]
            empty_support = pad_seq([], max_support_length, self.lexicon[self.pad])
            # support_seqs_padded = [self.pad_seq([self.pad_seq(s, max_support_length) for s in supports], max_num_support) for supports in support_seqs]
            support_seqs_padded = [
                pad_seq([pad_seq(s, max_support_length, self.lexicon[self.pad]) for s in batch_item],
                        max_num_support, empty_support)
                for batch_item in support_seqs]

            support_length = [[len(c) for c in pad_seq(inst, max_num_support, [])] for inst in support_seqs]


            question_seqs = [[self.lexicon[t]
                           for t in word_tokenize(inst['questions'][0]['question'])]
                         for inst in batch]

            max_question_length = self.all_max_question_length

            # [batch_size, max_question_length]
            question_seqs_padded = [pad_seq(q, max_question_length, self.lexicon[self.pad]) for q in question_seqs]

            neg_question_seqs_padded = []
            for qi in question_seqs_padded:
                rq = []
                while rq == [] or rq == qi:
                    rq = self.random.choice(self.all_question_seqs_padded)
                neg_question_seqs_padded.append(rq)


            target_values = [[transSentToIO(word_tokenize(support['text']), inst['questions'][0]['answers'])
                              for support in inst['support']]
                            for inst in batch]

            target_values_padded = [pad_seq([pad_seq(s, max_support_length, 0.0) for s in batch_item],
                        max_num_support, 0.0)
                for batch_item in target_values]


            #target_values_padded = [pad_seq(q, max_question_length, 0.0) for q in target_values] #[[c for c in pad_seq(inst, max_num_cands, 0.0)] for inst in target_values]
            #neg_target_values_padded = [[0.0 for c in q] for q in target_values_padded] #[pad_seq(q, max_question_length, 0.0) for q in target_values]

            question_length = [len(q) for q in question_seqs]
            #todo: change to actual lengths
            question_length_neg = [len(q) for q in neg_question_seqs_padded]

            # target values for test are not supplied, performance at test time is estimated by printing to converting back to quebaps again

            # sample negative candidate

            if test:
                yield {
                self.questions: question_seqs_padded,
                self.question_lengths: question_length,
                #self.candidates: candidate_seqs_padded,
                #self.candidate_lengths: candidate_length,
                self.support: support_seqs_padded,
                self.support_lengths: support_length
                }
            else:
                yield {
                self.questions: [(pos, neg) for pos, neg in zip(question_seqs_padded, neg_question_seqs_padded)],
                self.question_lengths: [(pos, neg) for pos, neg in zip(question_length, question_length_neg)],
                #self.candidates: candidate_seqs_padded,
                #self.candidate_lengths: candidate_length,
                self.target_values: target_values_padded,
                self.support: support_seqs_padded,
                self.support_lengths: support_length
                }



    def convert_to_predictions(self, batch, scores):
        """
        Convert a batched candidate tensor and batched scores back into a python dictionary in quebap format.
        :param candidates: candidate representation as generated by this tensorizer.
        :param scores: scores tensor of the shape of the target_value placeholder.
        :return: sequence of reading instances corresponding to the input.
        """
        # todo: update to work with current batcher
        theta = 0.5
        all_results = []
        for scores_per_question, supports_per_question in zip(scores, batch.suport):  # for inst in batch

            result_for_question = []
            all_preds = []
            all_scores = []

            for scores, support_seqs in zip(scores_per_question, supports_per_question):  # for each support seq

                curr_pred = []
                curr_sco = []
                len_sco = 0
                for sco, sup in zip(scores, support_seqs):
                    if (sco > theta and sup != self.lexicon[self.pad]):
                        curr_pred.append(sup)
                        len_sco += 1
                        curr_sco.append(sco)
                    elif (sco > theta and sup == self.lexicon[self.pad]):
                        curr_pred.append(self.pad)
                        len_sco += 1
                        curr_sco.append(sco)
                    else:
                        if len(curr_pred) > 0:
                            pred = " ".join(curr_pred)
                            all_preds.append(pred)
                            score = np.sum(curr_sco) / len_sco
                            all_scores.append(score)
                        curr_pred = []
                        len_sco = 0
                        curr_sco = []

                #answer_tokens = [self.lexicon.key_by_id(sup) for sco, sup in zip(scores, support_seqs) if
                #                 (sco > theta and sup != self.lexicon[self.pad])]

            for au in set(all_preds):
                indices = [i for i, x in enumerate(all_preds) if x == au]
                scores = [all_scores[i] for i in indices]

                total_score = np.sum(scores) / len(scores)

                candidate = {
                    'text': au,
                    'score': total_score
                }
                result_for_question.append(candidate)
            question = {'answers': sorted(result_for_question, key=lambda x: -x['score'])}
            quebap = {'questions': [question]}
            all_results.append(quebap)
        return all_results



