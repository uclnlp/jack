import tensorflow as tf
import numpy as np
import quebap.util.tfutil as tfutil


def reader(inputs, context, seq_lengths, repr_dim, vocab_size, emb_name="embedding_matrix", rnn_scope="RNN"):
    #Tim's suggestion: (inputs, context, indices):
    """
    todo: a reusable RNN based reader

    :param input: [batch_size x seq_length] input of int32 word ids
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