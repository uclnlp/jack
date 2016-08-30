import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn import *
from quebap.projects.autoread.wikireading.my_seq2seq import *
from quebap.projects.autoread.wikireading.qa import QASetting
from quebap.util import tfutil


class ProjectionWrapper(RNNCell):

    def __init__(self, cell, output_size, w, b, activation_fn=tf.nn.relu):
        self._cell = cell
        self._output_size = output_size
        self._w = w
        self._b = b

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell and output projection on inputs, starting from state."""
        output, res_state = self._cell(inputs, state)
        # Default scope: "OutputProjectionWrapper"
        with vs.variable_scope(scope or type(self).__name__):
            projected = tf.nn.relu(tf.matmul(output, self._w) + self._b)
        return projected, res_state


class QASeq2SeqModel:

    def __init__(self, size, vocab_size, answer_vocab_size,
                 max_context_length, max_question_length, max_answer_length,
                 beam_size=1, is_train=True, learning_rate=1e-2, keep_prob=1.0,
                 composition="GRU", devices=None, name="GatedSeq2SeqModel"):
        self._vocab_size = vocab_size
        self._max_context_length = max_context_length
        self._max_question_length = max_question_length
        self._max_answer_length = max_answer_length
        self._size = size
        self._is_train = is_train
        self._composition = composition
        self._beam_size = beam_size
        self._answer_vocab_size = answer_vocab_size
        self._device0 = devices[0] if devices is not None else "/cpu:0"
        self._device1 = devices[1 % len(devices)] if devices is not None else "/cpu:0"
        self._device2 = devices[2 % len(devices)] if devices is not None else "/cpu:0"

        if composition == "GRU":
            self._cell = GRUCell(self._size)
        else:
            self._cell = BasicLSTMCell(self._size)

        self._init = tf.random_normal_initializer(0.0, 0.1)
        with tf.device(self._device0):
            with tf.variable_scope(name, initializer=tf.contrib.layers.xavier_initializer()):
                self._init_inputs()
                self.keep_prob = tf.get_variable("keep_prob", [], initializer=tf.constant_initializer(keep_prob))
                with tf.variable_scope("embeddings"):
                    with tf.device("/cpu:0"):
                        # embeddings
                        self.answer_embeddings = tf.get_variable("E_answers", [answer_vocab_size, self._size],
                                                                 initializer=self._init)
                        self.input_embeddings = tf.get_variable("E_words", [vocab_size, self._size],
                                                                initializer=self._init)

                        self._batch_size = tf.shape(self._question)[0]
                        self._batch_size_32 = tf.squeeze(self._batch_size)

                        # [B, MAX_T, S] embedded question and context and answers
                        embedded_question, _ = self._embed(self._question, self._question_length, self.input_embeddings,
                                                           time_major=False)
                        embedded_context, _ = self._embed(self._context, self._context_length, self.input_embeddings,
                                                          time_major=False)
                        embedded_answers, answers = self._embed(self._answer, self._answer_length, self.answer_embeddings,
                                                                time_major=False)

                with tf.variable_scope("encoding"):
                    enc_question, enc_outs, enc_state = self.encoder(embedded_question, embedded_context)

                with tf.variable_scope("decoding"):
                    if enc_state is not None:
                        self.start_dec_state = tf.contrib.layers.fully_connected(tf.concat(1, [enc_state,
                                                                                               enc_question]),
                                                                                 self._size
                                                                                 if self._composition == "GRU" else
                                                                                 2 * self._size,
                                                                                 activation_fn=tf.tanh)
                    else:
                        self.start_dec_state = tf.contrib.layers.fully_connected(enc_question,
                                                                                 self._size
                                                                                 if self._composition == "GRU" else
                                                                                 2 * self._size,
                                                                                 activation_fn=tf.tanh)

                    self.start_dec_state = tf.gather(self.start_dec_state, self._answer_partition)

                    self.w_decode = tf.get_variable("w_decode", [self._size, self._answer_vocab_size])
                    self.b_decode = tf.get_variable("b_decode", [self._answer_vocab_size], tf.float32,
                                                    tf.constant_initializer(0.0))

                    outputs, self.decoder_symbols, self.beam_path, self.beam_symbols = \
                        self.decoder(self.start_dec_state, enc_outs, embedded_answers)

                self.model_params = [p for p in tf.trainable_variables() if name in p.name]

                if is_train:
                    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
                    self.global_step = tf.Variable(0, trainable=False, name="step")
                    self._opt = tf.train.AdamOptimizer(self.learning_rate)
                    # loss: [B * T]

                    #remove first answer_word and flatten answers to align with logits
                    answers = tf.slice(answers, [0, 1], [-1, -1])
                    flat_answers = tf.reshape(answers, [-1])
                    answer_mask = tf.slice(self._answer_mask, [0, 0], tf.shape(answers))

                    # [B * T, S]
                    outputs = tf.slice(outputs, [0, 0, 0], tf.pack([-1, tf.cast(tf.reduce_max(self._answer_length)-1, tf.int32),-1]))
                    self.logits = tf.nn.xw_plus_b(tf.reshape(outputs, [-1, self._size]), self.w_decode, self.b_decode)

                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, flat_answers)
                    loss = tf.reshape(loss, tf.shape(answer_mask)) * answer_mask
                    mean_loss = tf.reduce_sum(loss, reduction_indices=[1], keep_dims=False) / \
                                tf.reduce_sum(self._answer_mask, reduction_indices=[1], keep_dims=False)

                    mean_loss = tf.segment_mean(mean_loss, self._answer_partition)
                    self.loss = tf.reduce_mean(mean_loss)

                    self._grads = tf.gradients(self.loss, self.model_params, colocate_gradients_with_ops=True)

                    grads, _ = tf.clip_by_global_norm(self._grads, 5.0)
                    self.update = self._opt.apply_gradients(zip(grads, self.model_params),
                                                            global_step=self.global_step)

                    self.all_params = [p for p in tf.all_variables() if name in p.name]
                    self.all_saver = tf.train.Saver(self.all_params)

                self.model_saver = tf.train.Saver(self.model_params)

    def decoder(self, start_state, encoder_outputs, embedded_answers, with_attention=False):
        w_proj = tf.get_variable("w_proj", [self._size, self._size])
        b_proj = tf.get_variable("b_proj", [self._size], tf.float32, tf.constant_initializer(0.0))

        cell = ProjectionWrapper(self._cell, self._size, w_proj, b_proj)
        # decoder for eval
        decoder_inputs = tf.split(1, self._max_answer_length, self._answer)
        decoder_inputs = [tf.reshape(inp, [-1]) for inp in decoder_inputs]

        def decoder(decoder_inputs, initial_state, cell, num_symbols,
                    embedding_size, embedding=None, output_projection=None,
                    feed_previous=False,
                    update_embedding_for_previous=True, scope=None, beam_search=True, beam_size=10):
            if with_attention:
                return embedding_attention_decoder(decoder_inputs, initial_state, encoder_outputs, cell, num_symbols,
                                                   embedding_size, embedding=embedding,
                                                   output_projection=output_projection,
                                                   feed_previous=feed_previous,
                                                   update_embedding_for_previous=update_embedding_for_previous,
                                                   scope=scope,
                                                   beam_search=beam_search, beam_size=beam_size)
            else:
                return embedding_rnn_decoder(decoder_inputs, initial_state, cell, num_symbols,
                                             embedding_size, embedding=embedding,
                                             output_projection=output_projection,
                                             feed_previous=feed_previous,
                                             update_embedding_for_previous=update_embedding_for_previous,
                                             scope=scope,
                                             beam_search=beam_search, beam_size=beam_size)

        beam_path, beam_symbols = None, None
        if self._beam_size > 1:
            _, _, beam_path, beam_symbols = \
                decoder(decoder_inputs, self.start_dec_state,
                        cell, self._answer_vocab_size, self._size,
                        embedding=self.answer_embeddings,
                        output_projection=(self.w_decode, self.b_decode),
                        beam_search=True, beam_size=self._beam_size)

            decoder_symbols = [tf.reshape(tf.gather(beam_symbols[i],
                                                    tf.slice(beam_path[(i+1) % len(beam_symbols)], [0], [1])), [1])
                               for i in range(len(beam_symbols))]
        else:
            decoder_symbols, _ = \
                decoder(decoder_inputs, self.start_dec_state, cell, self._answer_vocab_size, self._size,
                        embedding=self.answer_embeddings,
                        output_projection=(self.w_decode, self.b_decode),
                        feed_previous=True,
                        beam_search=False)

        # Use dynamic rnn for training
        tf.get_variable_scope().reuse_variables()
        # [B, T, S]
        outputs, _ = dynamic_rnn(cell, embedded_answers, self._answer_length-1, self.start_dec_state,
                                 scope="embedding_rnn_decoder/rnn_decoder", time_major=False)

        return outputs, decoder_symbols, beam_path, beam_symbols

    def encoder(self, embedded_question, embedded_context):
        pass

    def set_train(self, sess):
        sess.run(self.keep_prob.initializer)

    def set_eval(self, sess):
        sess.run(self.keep_prob.assign(1.0))


    def _init_inputs(self):
        with tf.device("/cpu:0"):
            self._question = tf.placeholder(tf.int64, shape=[None, self._max_question_length], name="question")
            self._question_length = tf.placeholder(tf.int64, shape=[None], name="question_length")

            self._context = tf.placeholder(tf.int64, shape=[None, self._max_context_length], name="context")
            self._context_length = tf.placeholder(tf.int64, shape=[None], name="context_length")

            self._answer = tf.placeholder(tf.int64, shape=[None, self._max_answer_length], name="answer")
            self._answer_length = tf.placeholder(tf.int64, shape=[None], name="answer_length")
            # mask[i,j] = 1.0 if j < answer_length[i] else 0.0
            self._answer_mask = tf.placeholder(tf.float32, shape=[None, self._max_answer_length-1], name="answer_mask")

            # multiple answers are possible so we need a partition of answers to respective questions in batch
            self._answer_partition = tf.placeholder(tf.int64, shape=[None],
                                                    name="answer_partition")


    def _cut_at_length(self, encoded, length):
        """
        :param encoded: [T, B, S] encoded sequence
        :param length: assuming T is max(length)
        :return: [B, S]
        """
        encoded = tf.reshape(encoded, [-1, self._size])
        offset = tf.range(0, self._batch_size_32)
        to_gather = length * self._batch_size_32 + offset
        return tf.gather(encoded, to_gather)

    def _embed(self, input, length, e, time_major=True):
        max_length = tf.cast(tf.reduce_max(length), tf.int32)
        if time_major:
            input = tf.transpose(input)
            input = tf.slice(input, [0, 0], tf.pack([max_length, -1]))
        else:
            input = tf.slice(input, [0, 0], tf.pack([-1, max_length]))
        embedded = tf.nn.embedding_lookup(e, input)
        embedded = tf.nn.dropout(embedded, self.keep_prob)
        return embedded, input

    def _birnn_projected(self, embedded, length):
        """
        Encodes all embedded inputs with bi-rnn
        :return: encoded input
        """
        max_length = tf.cast(tf.reduce_max(length), tf.int32)

        with tf.device(self._device1):
            #use other device for backward rnn
            with tf.variable_scope("backward"):
                rev_embedded = tf.reverse_sequence(embedded, length, 1, 0)
                outs_bw = dynamic_rnn(self._cell, rev_embedded, length, dtype=tf.float32, time_major=False)[0]
                outs_bw = tf.reverse_sequence(outs_bw, length, 1, 0)
                out_bw = tf.reshape(outs_bw, [-1, self._size])

        with tf.device(self._device2):
            with tf.variable_scope("forward"):
                outs_fw = dynamic_rnn(self._cell, embedded, length, dtype=tf.float32, time_major=False)[0]
                out_fw = tf.reshape(outs_fw, [-1, self._size])
            # form query from forward and backward compositions

            encoded = tf.contrib.layers.fully_connected(tf.concat(1, [out_fw, out_bw]), self._size,
                                                        activation_fn=None, weights_initializer=None,
                                                        biases_initializer=None)

            encoded = tf.reshape(encoded, tf.pack([-1, max_length, self._size]))
            encoded = tf.add_n([encoded, outs_fw, outs_bw])

        return encoded

    def run(self, sess, goal, qa_settings):
        question = []
        question_length = []

        context = []
        context_length = []

        answer = []
        answer_length = []
        # mask[i,j] = 1.0 if j < answer_length[i] else 0.0
        answer_mask = []

        # multiple answers are possible so we need a partition of answers to respective questions in batch
        answer_partition = []

        for i, qa_setting in enumerate(qa_settings):
            question.append(qa_setting.question[:self._max_question_length] + \
                            [0] * (self._max_question_length - len(qa_setting.question)))
            question_length.append(len(qa_setting.question[:self._max_question_length]))
            context.append(qa_setting.context[:self._max_context_length] + \
                            [0] * (self._max_context_length - len(qa_setting.context)))
            context_length.append(len(qa_setting.context[:self._max_context_length]))
            for a in qa_setting.answers:
                answer.append(a + [0] * (self._max_answer_length-len(a)))
                answer_length.append(len(a))
                answer_partition.append(i)
                answer_mask.append([1.0] * (len(a)-1) + [0.0] * (self._max_answer_length-len(a)))

        feed_dict = {
            self._question: question,
            self._question_length: question_length,
            self._context: context,
            self._context_length: context_length,
            self._answer: answer,
            self._answer_length: answer_length,
            self._answer_mask: answer_mask,
            self._answer_partition: answer_partition
        }

        return sess.run(goal, feed_dict=feed_dict)


class AttentiveAnswerSeq2SeqModel(QASeq2SeqModel):

    def encoder(self, embedded_question, embedded_context):
        with tf.variable_scope("question"):
            # [B, S]
            encoded_question_for_select = dynamic_rnn(self._cell, embedded_question, self._question_length,
                                                            dtype=tf.float32, time_major=False)[1]
            max_length = tf.cast(tf.reduce_max(self._context_length), tf.int32)
            tiled_question = tf.expand_dims(encoded_question_for_select, 1)
            tiled_question = tf.tile(tiled_question, [1, max_length, 1])
            tiled_question.set_shape([None, None, self._size])

        with tf.variable_scope("question_decode"):
            encoded_question = dynamic_rnn(self._cell, embedded_question, self._question_length,
                                           dtype=tf.float32, time_major=False)[1]

        with tf.variable_scope("reverse_encoding"):
            #backward
            rev_embedded = tf.reverse_sequence(embedded_context, self._context_length, 1, 0)
            enc_rev_outputs, _ = dynamic_rnn(self._cell, rev_embedded, sequence_length=self._context_length,
                                             time_major=False, dtype=tf.float32)
            enc_rev_outputs = tf.reverse_sequence(enc_rev_outputs, self._context_length, 1, 0)

        with tf.variable_scope("encoding"):
            reading_input = tf.concat(2, [embedded_context, tiled_question])
            with tf.variable_scope("context"):
                # also encode reverse
                ctxt_input = tf.concat(2, [reading_input, enc_rev_outputs])
                ctxt_outputs, ctxt_state = dynamic_rnn(self._cell, ctxt_input, sequence_length=self._context_length,
                                                       time_major=False, dtype=tf.float32)

            with tf.variable_scope("potential_answer"):
                a_outputs, a_state = dynamic_rnn(self._cell, reading_input, sequence_length=self._context_length,
                                                 time_major=False, dtype=tf.float32)

        with tf.variable_scope("attention"):
            # [B, MAX_T, 1]
            weights = tf.batch_matmul(ctxt_outputs, tf.expand_dims(encoded_question_for_select, 2))
            # [B, MAX_T]
            weights = tf.reshape(weights, tf.pack([self._batch_size_32, -1]))
            # sharpen
            weights = weights * tf.get_variable("sharper", [], initializer=tf.constant_initializer(1.0))
            mask = tfutil.mask_for_lengths(self._context_length, max_length=max_length, batch_size=self._batch_size_32)
            self.weights = tf.nn.softmax(weights + mask)
            output = tf.reduce_sum(tf.expand_dims(self.weights, 2) * a_outputs, reduction_indices=[1])
            output.set_shape([None, self._size])

        return encoded_question, a_outputs, output


class AttentiveSeq2SeqModel(QASeq2SeqModel):

    def encoder(self, embedded_question, embedded_context):
        with tf.variable_scope("question"):
            # [B, S]
            encoded_question_for_select = dynamic_rnn(self._cell, embedded_question, self._question_length,
                                                            dtype=tf.float32, time_major=False)[1]
            max_length = tf.cast(tf.reduce_max(self._context_length), tf.int32)
            tiled_question = tf.expand_dims(encoded_question_for_select, 1)
            tiled_question = tf.tile(tiled_question, [1, max_length, 1])
            tiled_question.set_shape([None, None, self._size])

        with tf.variable_scope("question_decode"):
            encoded_question = dynamic_rnn(self._cell, embedded_question, self._question_length,
                                           dtype=tf.float32, time_major=False)[1]

        with tf.variable_scope("reverse_encoding"):
            #backward
            rev_embedded = tf.reverse_sequence(embedded_context, self._context_length, 1, 0)
            enc_rev_outputs, _ = dynamic_rnn(self._cell, rev_embedded, sequence_length=self._context_length,
                                             time_major=False, dtype=tf.float32)
            enc_rev_outputs = tf.reverse_sequence(enc_rev_outputs, self._context_length, 1, 0)

        with tf.variable_scope("encoding"):
            reading_input = tf.concat(2, [embedded_context, tiled_question, enc_rev_outputs])
            # also encode reverse
            ctxt_outputs, ctxt_state = dynamic_rnn(self._cell, reading_input, sequence_length=self._context_length,
                                                   time_major=False, dtype=tf.float32)

        with tf.variable_scope("attention"):
            # [B, MAX_T, 1]
            weights = tf.batch_matmul(ctxt_outputs, tf.expand_dims(encoded_question_for_select, 2))
            # [B, MAX_T]
            weights = tf.reshape(weights, tf.pack([self._batch_size_32, -1]))
            # sharpen
            weights = weights * tf.get_variable("sharper", [], initializer=tf.constant_initializer(1.0))
            mask = tfutil.mask_for_lengths(self._context_length, max_length=max_length, batch_size=self._batch_size_32)
            self.weights = tf.nn.softmax(weights + mask)
            output = tf.reduce_sum(tf.expand_dims(self.weights, 2) * ctxt_outputs, reduction_indices=[1])
            output.set_shape([None, self._size])

        return encoded_question, ctxt_outputs, output


class QConditionedSeq2SeqModel(QASeq2SeqModel):

    def encoder(self, embedded_question, embedded_context):
        with tf.variable_scope("question"):
            # [B, S]
            encoded_question_for_conditioning = dynamic_rnn(self._cell, embedded_question, self._question_length,
                                                            dtype=tf.float32, time_major=False)[1]

        with tf.variable_scope("question_decode"):
            encoded_question = dynamic_rnn(self._cell, embedded_question, self._question_length,
                                           dtype=tf.float32, time_major=False)[1]

        with tf.variable_scope("reverse_encoding"):
            #backward
            rev_embedded = tf.reverse_sequence(embedded_context, self._context_length, 1, 0)
            enc_rev_outputs, _ = dynamic_rnn(self._cell, rev_embedded, sequence_length=self._context_length,
                                             time_major=False, dtype=tf.float32)
            enc_rev_outputs = tf.reverse_sequence(enc_rev_outputs, self._context_length, 1, 0)
        with tf.variable_scope("encoding"):
            max_length = tf.cast(tf.reduce_max(self._context_length), tf.int32)
            tiled_question = tf.expand_dims(encoded_question_for_conditioning, 1)
            tiled_question = tf.tile(tiled_question, [1, max_length, 1])
            tiled_question.set_shape([None, None, self._size])
            reading_input = tf.concat(2, [embedded_context, enc_rev_outputs, tiled_question])

            enc_outputs, enc_state = dynamic_rnn(self._cell, reading_input, sequence_length=self._context_length,
                                                 time_major=False, dtype=tf.float32)

        return encoded_question, enc_outputs, enc_state


if __name__ == '__main__':
    import random
    qa_settings = [
        QASetting([1], [[0, 3, 2]], [1, 3, 2]),
        QASetting([1], [[0, 2, 3]], [1, 0, 2, 3]),
        QASetting([2], [[0, 1]], [0,2,1]),
        QASetting([2], [[0, 2], [0, 1], [0, 3]], [2,1,2,2,3])
    ]

    model = AttentiveSeq2SeqModel(size=10, vocab_size=4, answer_vocab_size=4, max_context_length=5,
                                     max_question_length=2, max_answer_length=3, beam_size=3)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for s in qa_settings:
            a = s.answers
            s.answers = [s.answers[0]]
            print(model.run(sess, model.decoder_symbols, [s]), [a1[1:] for a1 in a])
            s.answers = a

        print(model.run(sess, [model.loss], qa_settings))
        for i in range(100):
            print(model.run(sess, [model.loss, model.update], qa_settings))

        for s in qa_settings:
            a = s.answers
            s.answers = [s.answers[0]]
            print(model.run(sess, model.decoder_symbols, [s]), [a1[1:] for a1 in a])
