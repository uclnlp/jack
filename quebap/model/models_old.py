import tensorflow as tf
import numpy as np
import quebap.util.tfutil as tfutil
from quebap.tensorizer import Feature_Tensorizer, MultipleChoiceReader, \
    AtomicTensorizer, SequenceTensorizer, count_features
from quebap.util import tfutil as tfutil


def reader(inputs, context, seq_lengths, repr_dim, vocab_size, emb_name="embedding_matrix", rnn_scope="RNN"):
    #Tim's suggestion: (inputs, context, indices):
    """
    todo: a reusable RNN based reader

    :param inputs: [batch_size x seq_length] input of int32 word ids
    :param context: [batch_size x state_size] representation of context
      (e.g. previous paragraph representation)
    :param indices: [batch_size x num_indices] indices of output representations
      (e.g. sentence endings)
    :return: outputs [batch_size x num_indices x hidden_size] output
      representations
    """

    # initialise embedding matrix
    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, repr_dim], -0.1, 0.1),
                                   name=emb_name, trainable=True)


    # [batch_size, max_seq_length, input_size]
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, inputs)

    # dummy test to see if the embedding lookup is working
    # Reduce along dimension 1 (`n_input`) to get a single vector (row) per input example
    # embedding_aggregated = tf.reduce_sum(embedded_inputs, [1])

    # is this right?
    #batch_size, state_size = tf.unpack(tf.shape(context))
    #last_context_state = tfutil.get_by_index(context, state_size)
    last_context_state = tfutil.get_last(context)

    # initialise with state of context
    with tf.variable_scope(rnn_scope):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=repr_dim, state_is_tuple=True)
        # returning [batch_size, max_time, cell.output_size]
        outputs, last_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=seq_lengths,
            inputs=embedded_inputs,
            initial_state=last_context_state
        )

    return tfutil.get_by_index(outputs, seq_lengths)


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


def create_sequence_embedding(inputs, seq_lengths, repr_dim, vocab_size, emb_name, rnn_scope):
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

    with tf.variable_scope(rnn_scope):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=repr_dim, state_is_tuple=True)
        # returning [batch_size, max_time, cell.output_size]
        outputs, last_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=seq_lengths,
            inputs=embedded_inputs)

    return tfutil.get_by_index(outputs, seq_lengths)


def create_dot_product_scorer(question_encodings, candidate_encodings):
    """

    :param question_encodings: [batch_size, enc_dim] tensor of question representations
    :param candidate_encodings: [batch_size, num_candidates, enc_dim] tensor of candidate encodings
    :return: a [batch_size, num_candidate] tensor of scores for each candidate
    """
    return tf.reduce_sum(tf.expand_dims(question_encodings, 1) * candidate_encodings, 2)


def create_softmax_loss(scores, target_values):
    """

    :param scores: [batch_size, num_candidates] logit scores
    :param target_values: [batch_size, num_candidates] vector of 0/1 target values.
    :return: [batch_size] vector of losses (or single number of total loss).
    """
    return tf.nn.softmax_cross_entropy_with_logits(scores, target_values)

#@model(supports_support=True, name="asdasd", datasets=["blah","blah"])  #decorator
def create_log_linear_reader(reference_data, **options):
    """
    Create a log-linear reader, i.e. with a log-linear combination of text features.
    :param options: 'repr_dim', dimension of representation .
    :return: ModelLogLinear
    """

    tensorizer = Feature_Tensorizer(reference_data, eval(options['feature_type']))

    #here: [n_candidates = n_features]
    candidate_weights = create_dense_embedding(tensorizer.candidates, tensorizer.num_candidates, tensorizer.num_candidates)    #[n_candidates, n_features]
    features = tensorizer.feature_values   #[batchsize, n_features] = [batchsize, n_candidates]

    scores = create_dot_product_scorer(features, candidate_weights)
    loss = create_softmax_loss(scores, tensorizer.target_values)
    return MultipleChoiceReader(tensorizer, scores, loss)


def create_model_f_reader(reference_data, **options):
    """
    Create a ModelF reader.
    :param options: 'repr_dim', dimension of representation .
    :param reference_data: the data to determine the question / answer candidate symbols.
    :return: ModelF
    """
    tensorizer = AtomicTensorizer(reference_data)
    question_encoding = create_dense_embedding(tensorizer.questions, options['repr_dim'], tensorizer.num_questions)
    candidate_encoding = create_dense_embedding(tensorizer.candidates, options['repr_dim'], tensorizer.num_candidates)
    scores = create_dot_product_scorer(question_encoding, candidate_encoding)
    loss = create_softmax_loss(scores, tensorizer.target_values)
    return MultipleChoiceReader(tensorizer, scores, loss)


def create_bag_of_embeddings_reader(reference_data, **options):
    """
    A reader that creates sequence representations of the input reading instance, and then
    models each question and candidate as the sum of the embeddings of their tokens.
    :param reference_data: the reference training set that determines the vocabulary.
    :param options: repr_dim, candidate_split (used for tokenizing candidates), question_split
    :return: a MultipleChoiceReader.
    """
    tensorizer = SequenceTensorizer(reference_data,
                                 candidate_split=options['candidate_split'],
                                 question_split=options['question_split'])

    # get embeddings for each question token
    # [batch_size, max_question_length, repr_dim]
    question_embeddings = create_dense_embedding(tensorizer.questions, options['repr_dim'], tensorizer.num_questions_symbols)
    question_encoding = tf.reduce_sum(question_embeddings, 1)  # [batch_size, repr_dim]

    # [batch_size, num_candidates, max_question_length, repr_dim
    candidate_embeddings = create_dense_embedding(tensorizer.candidates, options['repr_dim'],
                                                  tensorizer.num_candidate_symbols)
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
    tensorizer = SequenceTensorizer(reference_data,
                                 candidate_split=options['candidate_split'],
                                 question_split=options['question_split'])

    # get embeddings for each question token
    # [batch_size, max_question_length, repr_dim]
    # inputs, seq_lengths, repr_dim, vocab_size

    question_encoding = create_sequence_embedding(tensorizer.questions, tensorizer.question_lengths, options['repr_dim'],
                                                  tensorizer.num_questions_symbols, "embedding_matrix_q", "RNN_q")

    # [batch_size, num_candidates, max_question_length, repr_dim
    candidate_embeddings = create_dense_embedding(tensorizer.candidates, options['repr_dim'],
                                                  tensorizer.num_candidate_symbols)
    candidate_encoding = tf.reduce_sum(candidate_embeddings, 2)  # [batch_size, num_candidates, repr_dim]
    scores = create_dot_product_scorer(question_encoding, candidate_encoding)
    loss = create_softmax_loss(scores, tensorizer.target_values)
    return MultipleChoiceReader(tensorizer, scores, loss)


def create_support_bag_of_embeddings_reader(reference_data, **options):
    """
    A reader that creates sequence representations of the input reading instance, and then
    models each question and candidate as the sum of the embeddings of their tokens.
    :param reference_data: the reference training set that determines the vocabulary.
    :param options: repr_dim, candidate_split (used for tokenizing candidates), question_split
    :return: a MultipleChoiceReader.
    """
    tensorizer = SequenceTensorizer(reference_data,
                                 candidate_split=options['candidate_split'],
                                 question_split=options['question_split'],
                                 support_split=options['support_split'])

    candidate_dim = options['repr_dim']
    support_dim = options['support_dim']

    # question embeddings: for each symbol a [support_dim, candidate_dim] matrix
    question_embeddings = tf.Variable(tf.random_normal((tensorizer.num_questions_symbols, support_dim, candidate_dim)))

    # [batch_size, max_question_length, support_dim, candidate_dim]
    question_encoding_raw = tf.gather(question_embeddings, tensorizer.questions)

    # question encoding should have shape: [batch_size, 1, support_dim, candidate_dim], so reduce and keep
    question_encoding = tf.reduce_sum(question_encoding_raw, 1, keep_dims=True)

    # candidate embeddings: for each symbol a [candidate_dim] vector
    candidate_embeddings = tf.Variable(tf.random_normal((tensorizer.num_candidate_symbols, candidate_dim)))
    # [batch_size, num_candidates, max_candidate_length, candidate_dim]
    candidate_encoding_raw = tf.gather(candidate_embeddings, tensorizer.candidates)

    # candidate embeddings should have shape: [batch_size, num_candidates, 1, candidate_dim]
    candidate_encoding = tf.reduce_sum(candidate_encoding_raw, 2, keep_dims=True)

    # each symbol has [support_dim] vector
    support_embeddings = tf.Variable(tf.random_normal((tensorizer.num_support_symbols, support_dim)))

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
