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
    embeddings = tf.Variable(tf.random_normal((num_symbols, repr_dim)))
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
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, repr_dim], -0.1, 0.1),
                                   name=emb_name, trainable=True)
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
            dtype=tf.float32,
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
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, repr_dim], -0.1, 0.1),
                                   name=emb_name, trainable=True)
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
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, repr_dim], -0.1, 0.1),
                                   name=emb_name, trainable=True)
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
            dtype=tf.float32,
            sequence_length=seq_lengths,
            inputs=embedded_inputs)


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
            dtype=tf.float32,
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
        # returning [batch_size, max_time, cell.output_size]
        outputs_fw_cond, last_state_fw_cond = tf.nn.dynamic_rnn(
            cell=cell_fw_cond,
            dtype=tf.float32,
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
        # outputs shape: [batch_size, max_time, cell.output_size]
        # last_states shape: [batch_size, cell.state_size]
        outputs_bw_cond, last_state_bw_cond = tf.nn.dynamic_rnn(
            cell=cell_fw,
            dtype=tf.float32,
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
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, repr_dim], -0.1, 0.1),
                                   name=emb_name, trainable=True)
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
            dtype=tf.float32,
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
            dtype=tf.float32,
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
            dtype=tf.float32,
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
            dtype=tf.float32,
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
    tensorizer = SequenceTensorizer(reference_data)

    dim1ql, dim2ql = tf.unpack(tf.shape(tensorizer.question_lengths))
    question_lengths_true = tf.squeeze(tf.slice(tensorizer.question_lengths, [0, 0], [dim1ql, 1]), [1])

    dim1q, dim2q, dim3q = tf.unpack(tf.shape(tensorizer.questions))
    questions_true = tf.squeeze(tf.slice(tensorizer.questions, [0, 0, 0], [dim1q, 1, dim3q]), [1])

    dim1t, dim2t, dim3t = tf.unpack(tf.shape(tensorizer.target_values))
    targets_true = tf.squeeze(tf.slice(tensorizer.target_values, [0, 0, 0], [dim1t, 1, dim3t]), [1])

    question_lengths_false = tf.squeeze(tf.slice(tensorizer.question_lengths, [0, 1], [dim1ql, 1]), [1])
    questions_false = tf.squeeze(tf.slice(tensorizer.questions, [0, 1, 0], [dim1q, 1, dim3q]), [1])
    targets_false = tf.squeeze(tf.slice(tensorizer.target_values, [0, 1, 0], [dim1t, 1, dim3t]), [1])


    # 1) bidirectional conditional encoding with one support
    question_encoding_true = create_bicond_question_encoding(tensorizer, questions_true, question_lengths_true, options, reuse_scope=False)
    question_encoding_false = create_bicond_question_encoding(tensorizer, questions_false, question_lengths_false, options, reuse_scope=True)
    cand_dim = options['repr_dim']*2

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
    #question_encoding_true = get_bicond_multisupport_question_encoding(tensorizer, questions_true, question_lengths_true, options, reuse_scope=False)
    #question_encoding_false = get_bicond_multisupport_question_encoding(tensorizer, questions_false, question_lengths_false, options, reuse_scope=True)
    #cand_dim = options['repr_dim'] * 2

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
    question_lengths_true = tf.squeeze(tf.slice(tensorizer.question_lengths, [0, 0], [dim1ql, 1]), [1])

    dim1q, dim2q, dim3q = tf.unpack(tf.shape(tensorizer.questions))
    questions_true = tf.squeeze(tf.slice(tensorizer.questions, [0, 0, 0], [dim1q, 1, dim3q]), [1])

    dim1t, dim2t, dim3t = tf.unpack(tf.shape(tensorizer.target_values))
    targets_true = tf.squeeze(tf.slice(tensorizer.target_values, [0, 0, 0], [dim1t, 1, dim3t]), [1])

    question_lengths_false = tf.squeeze(tf.slice(tensorizer.question_lengths, [0, 1], [dim1ql, 1]), [1])
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
    scores_all = scores_true + scores_false
    loss_all = loss_true + loss_false

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

    sup = tf.squeeze(tf.slice(tensorizer.support, [0, 0, 0], [dim1s, 1, dim3s]), [1])  # take the first support, this is the middle dimension
    sup_l = tf.squeeze(tf.slice(tensorizer.support_lengths, [0, 0], [dim1s, 1]), [1])

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
    sup = tf.reshape(tensorizer.support, [-1, dim3s])  # [batch_size * num_supports, num_tokens]

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
    question_embeddings = tf.Variable(tf.random_normal((tensorizer.num_symbols, support_dim, candidate_dim)))

    # [batch_size, max_question_length, support_dim, candidate_dim]
    question_encoding_raw = tf.gather(question_embeddings, tensorizer.questions)

    # question encoding should have shape: [batch_size, 1, support_dim, candidate_dim], so reduce and keep
    question_encoding = tf.reduce_sum(question_encoding_raw, 1, keep_dims=True)

    # candidate embeddings: for each symbol a [candidate_dim] vector
    candidate_embeddings = tf.Variable(tf.random_normal((tensorizer.num_symbols, candidate_dim)))
    # [batch_size, num_candidates, max_candidate_length, candidate_dim]
    candidate_encoding_raw = tf.gather(candidate_embeddings, tensorizer.candidates)

    # candidate embeddings should have shape: [batch_size, num_candidates, 1, candidate_dim]
    candidate_encoding = tf.reduce_sum(candidate_encoding_raw, 2, keep_dims=True)

    # each symbol has [support_dim] vector
    support_embeddings = tf.Variable(tf.random_normal((tensorizer.num_symbols, support_dim)))

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