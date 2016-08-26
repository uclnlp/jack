def reader(input, context, indices):
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
    pass
