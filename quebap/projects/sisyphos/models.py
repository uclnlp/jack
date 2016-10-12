import tensorflow as tf
import numpy as np

def embedder(inputs, input_size, vocab_size, embeddings=None, scope=None):
    """
    embeds the given input indices into a [batch_size, max_seq_length, input_size] tensor, based on the
    tensor embeddings [vocab_size, embedding_size]. Creates the embeddings tensor if needed.
    In case input_size differs from embedding_size, a linear transformation is performed.
    :param inputs: tensor with instance indices for the current batch (typically a placeholder)
    :param input_size: embedding length
    :param vocab_size: number of symbols in the vocabulary
    :param embeddings: tensor with shape [vocab_size, embedding_size], or None.
    :param scope: scope for the embedding matrix and embedded inputs
    :return: [batch_size, max_seq_length, input_size] tensor with embedded inputs
    """

    # todo: clean up - input arguments input_size, vocab_size not needed if embeddings is not None
    with tf.variable_scope(scope or "embedder") as varscope:

        # initializer = tf.random_normal_initializer(0.0, 0.1)
        initializer = tf.random_uniform_initializer(-0.05, 0.05)
        if embeddings is None:
            embeddings = \
                tf.get_variable("W", [vocab_size, input_size],
                                initializer=initializer)
            # [batch_size, max_seq_length, input_size]
            return tf.nn.embedding_lookup(embeddings, inputs)

        else:
            static_shape = tf.Tensor.get_shape(embeddings)
            if static_shape[1] == input_size:
                # [batch_size, max_seq_length, input_size]
                return tf.nn.embedding_lookup(embeddings, inputs)
            else:
                lin = tf.get_variable("W_trf", [1,static_shape[1], input_size],
                                    initializer=initializer)
                embedded = tf.nn.embedding_lookup(embeddings, inputs)
                lins = tf.tile(lin,[tf.shape(embedded)[0],1,1])
                rescaled_embedded = tf.batch_matmul(embedded,lins)
                #restore some shape information
                rescaled_embedded.set_shape([None, None, input_size])
                return rescaled_embedded






def create_embeddings(vocab, retrain=False, scope=None):
    """
    create embedding tensor with pre-trained embeddings
    :param vocab: instance of class VocabEmb
    :param retrain: False if pretrained embeddings are fixed, True otherwise
    :return: embeddings tensor with shape [vocab_size, embedding_length]
    """
    # todo: additional functionality to extend pretrained embeddings with non/trainable extra dimensions

    assert vocab.__class__.__name__=="VocabEmb", 'create_embeddings() needs VocabEmb instance'
    # todo: reduce VocabEmb to single Vocab class

    #embeddings = np.random.normal(loc=0.0, scale=0.1, size=[len(vocab), input_size]).astype("float32")

    v_shape = vocab.get_shape()
    embeddings = np.random.uniform(low=-0.05, high=0.05, size=[v_shape[0], v_shape[1]]).astype("float32")
    index_pretrained = vocab.get_normalized_ids_pretrained()
    syms_pretrained = vocab.get_syms_pretrained()
    for i,sym in zip(index_pretrained,syms_pretrained):
        vec = vocab.emb(sym)
        if vec is not None: #should not happen if same emb was used to create vocab
            embeddings[i] = vec

    with tf.variable_scope(scope or "embedder") as varscope:
        if retrain: #all are trainable
            E = tf.get_variable("W", initializer=tf.identity(embeddings), trainable=True)
        else:
            E_tune = tf.get_variable("W_tune", initializer=tf.identity(embeddings[:vocab.count_oov()]), trainable=True)
            E_fixed = tf.get_variable("W_fixed", initializer=tf.identity(embeddings[vocab.count_oov():]), trainable=False)
            E = tf.concat(0, [E_tune, E_fixed], name="W")
            #first out-of-vocab (tunable), then pre-trained, corresponding to normalized VocabEmb indices
    return E


def get_total_trainable_variables():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def get_total_variables():
    total_parameters = 0
    for variable in tf.all_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def reader(inputs, lengths, output_size, contexts=(None, None), scope=None):
    with tf.variable_scope(scope or "reader") as varscope:
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        _, (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            inputs,
            sequence_length=lengths,
            initial_state_fw=contexts[0],
            initial_state_bw=contexts[1],
            dtype=tf.float32
        )

        # each [batch_size x max_seq_length x output_size]
        return states_fw, states_bw


def conditional_reader(seq1, seq1_lengths, seq2, seq2_lengths, output_size, scope=None):
    with tf.variable_scope(scope or "conditional_reader_seq1") as varscope1:
        # (c_fw, h_fw), (c_bw, h_bw)
        seq1_states = \
            reader(seq1, seq1_lengths, output_size, scope=varscope1)
    #with tf.variable_scope(scope or "conditional_reader_seq2") as varscope2:
        varscope1.reuse_variables()
        # each [batch_size x max_seq_length x output_size]
        return reader(seq2, seq2_lengths, output_size, seq1_states, scope=varscope1)


def predictor(output, targets, target_size):
    logits = tf.contrib.layers.fully_connected(output, target_size)
    loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets))
    predict = tf.arg_max(tf.nn.softmax(logits), 1)
    return logits, loss, predict


def conditional_reader_model(input_size, output_size, vocab_size, target_size, embeddings=None):
    # Model
    # [batch_size, max_seq1_length]
    seq1 = tf.placeholder(tf.int64, [None, None], "seq1")
    # [batch_size]
    seq1_lengths = tf.placeholder(tf.int64, [None], "seq1_lengths")

    # [batch_size, max_seq2_length]
    seq2 = tf.placeholder(tf.int64, [None, None], "seq2")
    # [batch_size]
    seq2_lengths = tf.placeholder(tf.int64, [None], "seq2_lengths")

    # [batch_size]
    targets = tf.placeholder(tf.int64, [None], "targets")

    with tf.variable_scope("embedders") as varscope:
        seq1_embedded = embedder(seq1, input_size, vocab_size, embeddings=embeddings)
        varscope.reuse_variables()
        seq2_embedded = embedder(seq2, input_size, vocab_size, embeddings=embeddings)


    print('TRAINABLE VARIABLES (only embeddings): %d'%get_total_trainable_variables())


    output = conditional_reader(seq1_embedded, seq1_lengths,
                                seq2_embedded, seq2_lengths,
                                output_size)

    output = tf.concat(1, [output[0][1], output[1][1]])

    logits, loss, predict = predictor(output, targets, target_size)

    print('TRAINABLE VARIABLES (embeddings + model): %d'%get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d'%get_total_variables())


    return (logits, loss, predict), \
           (seq1, seq1_lengths, seq2, seq2_lengths, targets)  # placeholders
