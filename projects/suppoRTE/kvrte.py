# -*- coding: utf-8 -*-

import tensorflow as tf

from jtr.nn.models import get_total_trainable_variables, get_total_variables, predictor


def key_value_reader(inputs, lengths, output_size, contexts=(None, None),
                     scope=None, drop_keep_prob=1.0, project_fw_bw=True):
    with tf.variable_scope(scope or "key_value_reader") as varscope:
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        if drop_keep_prob != 1.0:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=cell,
                output_keep_prob=drop_keep_prob
            )

        # [batch_size x seq_length x output_size], ?
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell,
            cell,
            inputs,
            sequence_length=lengths,
            initial_state_fw=contexts[0],
            initial_state_bw=contexts[1],
            dtype=tf.float32
        )

        outputs_fw, outputs_bw = outputs

        outputs_fw_key, outputs_fw_val = tf.split(2, 2, outputs_fw)
        outputs_bw_key, outputs_bw_val = tf.split(2, 2, outputs_bw)
        outputs_key = tf.concat(2, [outputs_fw_key, outputs_bw_key])
        outputs_val = tf.concat(2, [outputs_fw_val, outputs_bw_val])

        if project_fw_bw:
            outputs_key = tf.contrib.layers.fully_connected(
                outputs_key, output_size, activation_fn=tf.tanh)
            outputs_val = tf.contrib.layers.fully_connected(
                outputs_val, output_size, activation_fn=tf.tanh)

        # outputs_key/outputs_val: [batch_size x max_length x output_size]
        return (outputs_key, outputs_val), states


def mutable_attention(memory_states, input, input_lengths,
                      output_size, scope=None):
    with tf.variable_scope(scope or "mutable_attention") as varscope1:
        batch_size = tf.shape(input)[0]
        max_time = tf.shape(input)[1]
        input_depth = int(input.get_shape()[2])

        # transforming input to time major
        input_major = tf.transpose(input, [1, 0, 2])
        num_units = output_size

        # fixme: very hacky and costly way
        input_lengths_cast = tf.cast(input_lengths, tf.int32)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unpack(input_major)

        # attention controller
        cell = tf.nn.rnn_cell.LSTMCell(num_units)

        #attention_states_fw, attention_states_bw = tf.split(0, 2, memory_states)
        #attention_states = tf.concat(3, [attention_states_fw, attention_states_bw])
        #attention_states = tf.squeeze(attention_states, [0])

        memory_key, memory_val = memory_states

        # transforming attention states time major
        memory_key_major = tf.transpose(memory_key, [1, 0, 2])
        memory_val_major = tf.transpose(memory_val, [1, 0, 2])

        attention_states = tf.contrib.layers.linear(memory_key_major, num_units)

        att_len = tf.shape(attention_states)[0]

        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:  # time == 0
                next_cell_state = cell.zero_state(batch_size, tf.float32)
            else:
                next_cell_state = cell_state
            elements_finished = (time >= input_lengths_cast)

            c, query = next_cell_state

            ## Working with memory keys
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

            ## Working with memory vals
            attention_states_flat = tf.reshape(memory_val_major, [-1, num_units])

            alpha_flat = tf.reshape(alpha, [-1, 1])

            # todo: so far only read operation! also use write operation!

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


def key_value_rte(placeholders, nvocab, **options):
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
    premise = placeholders['question']
    # [batch_size]
    premise_lengths = placeholders["question_lengths"]
    # [batch_size, max_seq2_length]
    hypothesis = placeholders["support"]
    # [batch_size]
    hypothesis_lengths = placeholders["support_lengths"]
    # [batch_size]
    targets = placeholders["answers"]

    output_size = options["repr_dim_output"]

    with tf.variable_scope("embedders") as varscope:
        premise_embedded = nvocab(premise)
        varscope.reuse_variables()
        hypothesis_embedded = nvocab(hypothesis)

    # todo: add option for attentive reader

    print('TRAINABLE VARIABLES (only embeddings): %d' % get_total_trainable_variables())

    with tf.variable_scope("key_value_readers") as varscope:
        premise_outputs, _ = \
            key_value_reader(premise_embedded, premise_lengths, output_size,
                             contexts=(None, None), scope=None,
                             drop_keep_prob=options["drop_keep_prob"],
                             project_fw_bw=True)
        varscope.reuse_variables()
        hypothesis_outputs, _ = \
            key_value_reader(hypothesis_embedded, hypothesis_lengths, output_size,
                             contexts=(None, None), scope=None,
                             drop_keep_prob=options["drop_keep_prob"],
                             project_fw_bw=True)

    # Reading premise with memory of hypothesis
    #premise_outputs_processed, premise_state = mutable_attention(
    #    hypothesis_outputs, premise_outputs[0], premise_lengths, output_size, scope=varscope)
    #varscope.reuse_variables()

    # Reading hypothesis with memory of premise and altered memory of hypothesis
    hypothesis_outputs_processed, hypothesis_state = mutable_attention(
        premise_outputs, hypothesis_outputs[0], hypothesis_lengths, output_size)

    # todo: read premise and hypothesis memory for inferring entailment class

    # fixme: last can be zero because of dynamic_rnn!
    #output = hypothesis_outputs[0][:, -1, :]

    output = hypothesis_state.h

    #targets = tf.Print(targets, [tf.shape(targets)], "targets ")
    #output = tf.Print(output, [tf.shape(output)], "outputs ")

    logits, loss, predict = predictor(output, targets, options["answer_size"])

    print('TRAINABLE VARIABLES (embeddings + model): %d' % get_total_trainable_variables())
    print('ALL VARIABLES (embeddings + model): %d' % get_total_variables())

    return logits, loss, predict
