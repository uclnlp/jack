import tensorflow as tf



from jtr.util import tfutil



def basic_rnn(max_length, lengths, ids, cell, E=None, additional_inputs=None, rev=False, init_state=None):
    """
    :param max_length: max. sequence length
    :param lengths: sequence lengths (tensor of length batch_size)
    :param ids: symbol ID's in tensor with shape (length x batch_size). 
           Note: this shape assumes time_major == True; inputs for rnn will have shape: `[max_time, batch_size, input_size]`
    :param cell: RNNCell
    :param E: None, or embedding matrix 
    :param additional_inputs: None, or additional inputs with shape (max seq length x batch size x embedding length)
    :param rev: (boolean) reverse sequences (default False)
    :param init_state: None, or initial state. Note: assume `cell.state_size` is a tuple => init_state should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    """
    
    inp = None
    if ids is not None:
        #with tf.device("/cpu:0"):
        inp = tf.nn.embedding_lookup(E, ids) # inp = input tensor (max seq length x batch size x embedding length)
        if additional_inputs is not None:#extend already given embeddings with the additional inputs (along dim 2 = embedding length)
            inp = tf.concat([inp, additional_inputs], 2) 
    else:
        inp = additional_inputs

    
    if rev:
        inp = tf.reverse_sequence(inp, lengths, 0, 1)  #seq_dim: 0, batch_dim: 1
        
    outputs, final_state = tf.nn.dynamic_rnn(cell, inp, sequence_length=lengths, initial_state=init_state, time_major=True, dtype=tf.float32)

    if rev: 
        outputs = tf.reverse_sequence(outputs, lengths, 0, 1)

    max_length = tf.gather(tf.shape(outputs), [0])   
    batch_size = tf.gather(tf.shape(lengths), [0])

#    flat_index = tf.range(0, max_length) * batch_size + (lengths - 1)
#    flat_index = tf.add(tf.mul(batch_size, tf.range(0, max_length)), tf.sub(lengths, 1))   
#    final_output = tf.gather(tf.reshape(outputs, [-1, cell.output_size]), flat_index) #shape: max_length x batch_size
    """Note:
    final_output = tfutil.get_by_index(outputs, lengths)  
    does not work, as it needs a numerical value for the 3rd dimension 
    (else: None in that direction, and fully_connected does not work)
    """

    mask = tf.expand_dims(tf.one_hot(lengths,max_length,axis=0,dtype=tf.int32),2) #seqlength x batch_size x 1
    mask = tf.cast(tf.tile(mask,tf.stack([1,1,cell.output_size])),tf.float32) #seqlength x batch_size x cell.output_size
    final_output = tf.reduce_sum(tf.mul(outputs,mask),0)
    
    return final_output, final_state, outputs
    


def create_embeddings(tunable_embeddings,fixed_embeddings):
    #with tf.device("/cpu:0"):
    E = None
    if fixed_embeddings is not None and fixed_embeddings.shape[0] > 0:
        E = tf.get_variable("E_fix", initializer=tf.identity(fixed_embeddings), trainable=True)
        #TODO: ask Dirk why like this, and afterwards explicitly removed from update
    if tunable_embeddings is not None and tunable_embeddings.shape[0] > 0:
        E_tune = tf.get_variable("E_tune", initializer=tf.identity(tunable_embeddings), trainable=True)
        if E is not None:
            E = tf.concat([E_tune, E], 0)
        else:
            E = E_tune
    return E #



def rte_model(max_length, l2_lambda, learning_rate, h_size, cellA, cellB, tunable_embeddings, fixed_embeddings, keep_prob,
                 initializer=tf.random_uniform_initializer(-0.05, 0.05)):
    """baseline rte model (based on Dirk Weissenborn's train_snli)
    :param max_length: max. sequence length
    :param l2_lambda
    :param learning_rate
    :param h_size
    :param cellA: instance of RNNCell (e.g. BasicLSTMCell, GRUCell)
    :param cellB    
    :param tunable_embeddings: ndarray with tunable embeddings
    :param fixed_embeddings: ndarray with fixed embeddings (None of all are tunable)
    :param keep_prob: keep prob for dropout
    :param initializer
    """
    
    with tf.variable_scope("model", initializer=initializer):#below: this initializer is used by default, except where overruled by explicit initializer 
        idsA = tf.placeholder(tf.int32, [max_length, None]) #return placeholder (max sequence length x batch size)
        idsB = tf.placeholder(tf.int32, [max_length, None]) #return placeholder (max sequence length x batch size)
        lengthsA = tf.placeholder(tf.int32, [None]) #return placeholder
        lengthsB = tf.placeholder(tf.int32, [None]) #return placeholder
        learning_rate = tf.get_variable("lr", (), tf.float32, tf.constant_initializer(learning_rate), trainable=False)

        batch_size = tf.gather(tf.shape(idsA), [1])

        keep_prob_var = tf.get_variable("keep_prob", (), initializer=tf.constant_initializer(keep_prob, tf.float32),
                                        trainable=False) #return op



        if keep_prob < 1.0:
            cellA = tf.nn.rnn_cell.DropoutWrapper(cellA, keep_prob_var, keep_prob_var)#input_keep_prob and output_keep_prob
            cellB = tf.nn.rnn_cell.DropoutWrapper(cellB, keep_prob_var, keep_prob_var)

        E = create_embeddings(tunable_embeddings,fixed_embeddings)
        
        with tf.variable_scope("rnn", initializer=initializer):
            premise_out, premise_state, premise_outs = basic_rnn(max_length, lengthsA, idsA, cellA, E)
            tf.get_variable_scope().reuse_variables()
#            if cellB.state_size > cellA.state_size:
#                rest_state = tf.zeros([cellB.state_size - cellA.state_size], tf.float32)
#                rest_state = tf.reshape(tf.tile(rest_state, batch_size), [-1, cellB.state_size - cellA.state_size])
#                s = tf.concat([rest_state, s], 1)
            #initial state: conditioned on premise state
            hypothesis_out, _, hypothesis_outs = basic_rnn(max_length, lengthsB, idsB, cellB, E, init_state=premise_state)

        h = tf.concat([premise_out, hypothesis_out], 1)
        h = tf.contrib.layers.fully_connected(h, h_size, activation_fn=tf.nn.relu)                

        #h = tf.concat([p, h, tf.abs(p-h)], 1)
        #h = tf.contrib.layers.fully_connected(h, h_size, activation_fn=lambda x: tf.maximum(0.0, x), weight_init=None)

        #calculate loss
        y = tf.placeholder(tf.int64, [None]) #return placeholder
        scores = tf.contrib.layers.fully_connected(h, 3) #return op
        loss =
        tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores,
            labels=labels)) / tf.cast(batch_size, tf.float32)

        train_params = tf.trainable_variables()#returns list of all variables with trainable=True
        if l2_lambda > 0.0:
            l2_loss = l2_lambda * tf.reduce_sum(tf.stack([tf.nn.l2_loss(t) for t in train_params]))
            loss = loss+l2_loss #return op

        probs = tf.nn.softmax(scores) #return op


    grads = tf.gradients(loss, train_params) 
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    grads_params = list(zip(grads, train_params))
    grads_params_ex_emb = [(g,p) for (g,p) in grads_params if not p.name.endswith("E_fix")]

    update = tf.train.AdamOptimizer(learning_rate, beta1=0.0).apply_gradients(grads_params) #return op
    update_exclude_embeddings = tf.train.AdamOptimizer(learning_rate, beta1=0.0).apply_gradients(grads_params_ex_emb) #return op
    return {"idsA":idsA, "idsB":idsB, "lengthsA":lengthsA, "lengthsB":lengthsB, "y":y,
            "probs":probs, "scores":scores,"keep_prob": keep_prob_var,
            "loss":loss, "update":update, "update_ex":update_exclude_embeddings}
