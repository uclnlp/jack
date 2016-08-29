from tensorflow.python.ops.rnn_cell import GRUCell, BasicLSTMCell
import tensorflow as tf

from quebap.util import tfutil


class AutoReader():
    def __init__(self, size, vocab_size, max_context_length,
                 is_train=True, learning_rate=1e-2, keep_prob=1.0,
                 composition="GRU", devices=None, name="AutoReader"):
        self._vocab_size = vocab_size
        self._max_context_length = max_context_length
        self._size = size
        self._is_train = is_train
        self._composition = composition
        self._device0 = devices[0] if devices is not None else "/cpu:0"
        self._device1 = devices[1 % len(devices)] if devices is not None else "/cpu:0"

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
                        self._batch_size = tf.shape(self._inputs)[0]
                        self._batch_size_32 = tf.squeeze(self._batch_size)

                        # [B, MAX_T, S] embedded context
                        embedded_context = self.embedder()

                with tf.variable_scope("encoding"):
                    self.outputs = self._birnn_projected(embedded_context)

                self.model_params = [p for p in tf.trainable_variables() if name in p.name]

                if is_train:
                    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
                    self.global_step = tf.Variable(0, trainable=False, name="step")
                    self._opt = tf.train.AdamOptimizer(self.learning_rate)
                    # loss: [B * T]

                    # remove first answer_word and flatten answers to align with logits
                    self.logits = self.symbolizer(self.outputs)
                    self.symbols = tf.arg_max(self.logits, 2)
                    self.loss = self.unsupervised_loss(self.logits)

                    self._grads = tf.gradients(self.loss, self.model_params, colocate_gradients_with_ops=True)
                    grads, _ = tf.clip_by_global_norm(self._grads, 5.0)
                    self.update = self._opt.apply_gradients(zip(grads, self.model_params),
                                                            global_step=self.global_step)

                    self.all_params = [p for p in tf.all_variables() if name in p.name]
                    self.all_saver = tf.train.Saver(self.all_params)

                self.model_saver = tf.train.Saver(self.model_params)

    def _init_inputs(self):
        with tf.device("/cpu:0"):
            self._inputs = tf.placeholder(tf.int64, shape=[None, self._max_context_length], name="context")
            self._seq_lengths = tf.placeholder(tf.int64, shape=[None], name="context_length")

    def _noiserizer(self, inputs):
        return tf.nn.dropout(inputs, self.keep_prob)

    def embedder(self):
        """
        :param inputs:
        :param input_size:
        :return:
        """
        with tf.variable_scope("embedder",
                               initializer=tf.random_normal_initializer()):
            self.input_embeddings = \
                tf.get_variable("embedding_matrix", shape=(self._vocab_size, self._size),
                                trainable=True)

        # [batch_size x max_seq_length x input_size]
        embedded_inputs = tf.nn.embedding_lookup(self.input_embeddings, self._inputs)

        return self._noiserizer(embedded_inputs)



    def _birnn_projected(self, embedded):
        """
        Encodes all embedded inputs with bi-rnn
        :return: [B, T, S] encoded input
        """
        max_length = tf.cast(tf.reduce_max(self._seq_lengths), tf.int32)

        with tf.device(self._device1):
            # use other device for backward rnn
            with tf.variable_scope("backward"):
                rev_embedded = tf.reverse_sequence(embedded, self._seq_lengths, 1, 0)
                outs_bw = tf.nn.dynamic_rnn(self._cell, rev_embedded, self._seq_lengths, dtype=tf.float32, time_major=False)[0]
                outs_bw = tf.reverse_sequence(outs_bw, self._seq_lengths, 1, 0)
                out_bw = tf.reshape(outs_bw, [-1, self._size])

        with tf.device(self._device0):
            with tf.variable_scope("forward"):
                outs_fw = tf.nn.dynamic_rnn(self._cell, embedded, self._seq_lengths, dtype=tf.float32, time_major=False)[0]
                out_fw = tf.reshape(outs_fw, [-1, self._size])
            # form query from forward and backward compositions

            encoded = tf.contrib.layers.fully_connected(tf.concat(1, [out_fw, out_bw]), self._size,
                                                        activation_fn=tf.tanh, weights_initializer=None,
                                                        biases_initializer=None)

            encoded = tf.reshape(encoded, tf.pack([-1, max_length, self._size]))
            encoded = tf.add_n([encoded, outs_fw, outs_bw])

        #[B, T, S]
        return encoded

    def symbolizer(self, outputs):
        """
        :param outputs: [batch_size x max_seq_length x output_dim]
        :return:
        """
        return tf.contrib.layers.fully_connected(outputs, self._vocab_size,
                                                 activation_fn=None)

    def unsupervised_loss(self, logits):
        """
        :param logits: [batch_size x max_seq_length x vocab_size]
        :return:
        """
        mask = tfutil.mask_for_lengths(self._seq_lengths, mask_right=False, value=1.0)
        mask_reshaped = tf.reshape(mask, shape=(-1,))
        logits_reshaped = tf.reshape(logits, shape=(-1, self._vocab_size))
        targets_reshaped = tf.reshape(self._inputs, shape=(-1,))

        # return tf.nn.softmax_cross_entropy_with_logits(masked_logits, targets)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits_reshaped, targets_reshaped
        )
        loss_masked = loss * mask_reshaped
        return tf.reduce_sum(loss_masked) / tf.reduce_sum(mask_reshaped)


    def run(self, sess, goal, batch):
        feed_dict = {
            self._inputs: batch[0],
            self._seq_lengths:  batch[1]
        }

        return sess.run(goal, feed_dict=feed_dict)