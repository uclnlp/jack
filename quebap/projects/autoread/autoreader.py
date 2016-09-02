"""
            __                       __
 ___ ___ __/ /____  _______ ___ ____/ /
/ _ `/ // / __/ _ \/ __/ -_) _ `/ _  /
\_,_/\_,_/\__/\___/_/  \__/\_,_/\_,_/ v0.1

Learning to read, unsupervised
"""

from tensorflow.python.ops.rnn_cell import GRUCell, BasicLSTMCell, RNNCell
import tensorflow as tf

from quebap.util import tfutil


class ParallelInputRNNCell(RNNCell):
    def __init__(self, cell):
        self._cell = cell

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    @property
    def state_size(self):
        return self._cell.state_size

    def __call__(self, inputs, state, scope=None):
        # input1: without noise
        # input2: with noise
        input1, input2 = tf.split(1, 2, inputs)

        # state of forward without noise
        _, new_state = self._cell(input1, state, scope)
        tf.get_variable_scope().reuse_variables()
        # output of forward with noise
        new_output, _ = self._cell(input2, state, scope)
        return new_output, new_state

    @property
    def output_size(self):
        return self._cell.output_size


class AutoReader():
    def __init__(self, size, vocab_size,
                 is_train=True, learning_rate=1e-2, dropout=1.0, cloze_noise=0.0,
                 composition="GRU", devices=None, name="AutoReader", unk_id=-1,
                 forward_only=False):
        self.unk_mask = None
        self.vocab_size = vocab_size
        self.size = size
        self._composition = composition
        self._device0 = devices[0] if devices is not None else "/cpu:0"
        self._device1 = devices[1 % len(devices)] if devices is not None else "/cpu:0"
        self._is_train = is_train
        self.unk_id = unk_id
        self.forward_only = forward_only

        if composition == "GRU":
            self._cell = GRUCell(self.size)
        else:
            self._cell = BasicLSTMCell(self.size)

        self._init = tf.random_normal_initializer(0.0, 0.1)
        with tf.device(self._device0):
            with tf.variable_scope(name, initializer=tf.contrib.layers.xavier_initializer()):
                self._init_inputs()
                self.keep_prob = tf.get_variable("keep_prob", [], initializer=tf.constant_initializer(1.0-dropout), trainable=False)
                self.cloze_noise = tf.get_variable("train_noise", [], initializer=tf.constant_initializer(cloze_noise), trainable=False)
                self.max_noise = tf.get_variable("max_noise", [], initializer=tf.constant_initializer(1.0), trainable=False)

                with tf.variable_scope("embeddings"):
                    with tf.device("/cpu:0"):
                        self.input_embeddings = \
                            tf.get_variable("embedding_matrix", shape=(self.vocab_size, self.size),
                                            initializer=self._init, trainable=True)

                        max_length = tf.cast(tf.reduce_max(self._seq_lengths), tf.int32)
                        inputs = tf.slice(self._inputs, (0, 0), tf.pack((-1, max_length)))

                        self._batch_size = tf.shape(self._inputs)[0]
                        self._batch_size_32 = tf.squeeze(self._batch_size)

                with tf.variable_scope("encoding"):
                    self.outputs = self._birnn_projected(inputs)

                self.model_params = [p for p in tf.trainable_variables() if name in p.name]

                self.logits = self.symbolizer(self.outputs)
                self.symbols = tf.arg_max(self.logits, 2)

                if is_train:
                    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
                    self.global_step = tf.Variable(0, trainable=False, name="step")
                    self._opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.0)
                    # loss: [B * T]

                    # remove first answer_word and flatten answers to align with logits
                    self.loss = self.unsupervised_loss(self.logits, inputs)
                    self._grads = tf.gradients(self.loss, self.model_params, colocate_gradients_with_ops=True)
                    grads, _ = tf.clip_by_global_norm(self._grads, 5.0)
                    self.update = self._opt.apply_gradients(zip(grads, self.model_params),
                                                            global_step=self.global_step)

                    self.all_params = [p for p in tf.all_variables() if name in p.name]
                    self.all_saver = tf.train.Saver(self.all_params, max_to_keep=2)

                self.model_saver = tf.train.Saver(self.model_params, max_to_keep=2)

    def _init_inputs(self):
        with tf.device("/cpu:0"):
            self._inputs = tf.placeholder(tf.int64, shape=[None, None], name="context")
            self._seq_lengths = tf.placeholder(tf.int64, shape=[None], name="context_length")
            self._weights = tf.placeholder(tf.float32, shape=[None, None], name="context_weigths")

    def _noiserizer(self, inputs, noise):
        return tf.cond(tf.equal(noise, 1.0),
                       lambda: tf.zeros_like(inputs),
                       lambda: tf.nn.dropout(inputs, 1-noise) * (1-noise))

    def _birnn_projected(self, inputs):
        """
        Encodes all embedded inputs with bi-rnn, up to max(self._seq_lengths)
        :return: [B, T, S] encoded input
        """
        #cell = ParallelInputRNNCell(self._cell) if self._is_train else self._cell

        cell = self._cell

        max_length = tf.cast(tf.reduce_max(self._seq_lengths), tf.int32)
        with tf.variable_scope("embedder", initializer=tf.random_normal_initializer()):
            # [batch_size x max_seq_length x input_size]
            embedded_inputs = tf.nn.embedding_lookup(self.input_embeddings, inputs)
            embedded = tf.nn.dropout(embedded_inputs, self.keep_prob)

            #if self._is_train:
            #    # (normal input, noisy input)
            #    embedded = tf.concat(2, [embedded, self._noiserizer(embedded_inputs, self.cloze_noise)])

            noise = tf.random_uniform([], self.cloze_noise, self.max_noise)

            cloze_embedding = tf.reshape(self._noiserizer(embedded_inputs, noise), [-1, self.size])

        with tf.device(self._device0):
            with tf.variable_scope("forward"):
                init_state_fw = tf.expand_dims(cell.zero_state(self._batch_size, dtype=tf.float32), 1)

                outs_fw_tmp = tf.nn.dynamic_rnn(cell, embedded, self._seq_lengths, dtype=tf.float32, time_major=False)[0]

                outs_fw = tf.slice(tf.concat(1, [init_state_fw, outs_fw_tmp]),
                                   [0, 0, 0], tf.pack([-1, max_length, -1]))
                out_fw = tf.reshape(outs_fw, [-1, self.size])

                if self.forward_only:
                    encoded = tf.contrib.layers.fully_connected(
                        tf.concat(1, [out_fw, cloze_embedding]),
                        self.size,
                        weights_initializer=None
                    )

                    encoded = tf.reshape(encoded, tf.pack([-1, max_length, self.size]))
                    encoded.set_shape((None, None, self.size))

                    return encoded

        with tf.device(self._device1):
            # use other device for backward rnn
            with tf.variable_scope("backward"):
                init_state_bw = tf.expand_dims(cell.zero_state(self._batch_size, dtype=tf.float32), 1)

                rev_embedded = tf.reverse_sequence(embedded, self._seq_lengths, 1, 0)
                outs_bw_tmp = tf.nn.dynamic_rnn(cell, rev_embedded, self._seq_lengths, dtype=tf.float32, time_major=False)[0]

                outs_bw = tf.slice(tf.concat(1, [init_state_bw, outs_bw_tmp]),
                                   [0, 0, 0], tf.pack([-1, max_length, -1]))

                outs_bw = tf.reverse_sequence(outs_bw, self._seq_lengths, 1, 0)
                out_bw = tf.reshape(outs_bw, [-1, self.size])

            encoded = tf.contrib.layers.fully_connected(
                tf.concat(1, [out_fw, out_bw, cloze_embedding]), self.size,
                weights_initializer=None
            )

            encoded = tf.reshape(encoded, tf.pack([-1, max_length, self.size]))
            #encoded = tf.add_n([encoded, outs_fw, outs_bw])

        #[B, T, S]
        encoded.set_shape((None, None, self.size))
        return encoded

    def symbolizer(self, outputs):
        """
        :param outputs: [batch_size * max_seq_length x output_dim]
        :return:
        """
        return tf.contrib.layers.fully_connected(outputs, self.vocab_size,
                                                 activation_fn=None)

    def unsupervised_loss(self, logits, targets):
        """
        :param logits: [batch_size * max_seq_length x vocab_size]
        :return:
        """
        mask = tfutil.mask_for_lengths(self._seq_lengths, mask_right=False, value=1.0)
        mask_reshaped = tf.reshape(mask, shape=(-1,))
        logits_reshaped = tf.reshape(logits, shape=(-1, self.vocab_size))
        targets_reshaped = tf.reshape(targets, shape=(-1,))
        mask_unk = tf.cast(tf.not_equal(tf.cast(self.unk_id, tf.int64), targets_reshaped), tf.float32)
        mask_final = mask_reshaped * mask_unk

        self.unk_mask = mask_unk

        weights = tf.slice(self._weights, [0, 0], tf.shape(targets))
        weights_reshaped = tf.reshape(weights, shape=(-1,))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_reshaped, targets_reshaped)
        loss_masked = loss * mask_final * weights_reshaped

        return tf.reduce_sum(loss_masked) / tf.reduce_sum(mask_final * weights_reshaped)

    def run(self, sess, goal, batch):
        feed_dict = {
            self._inputs: batch[0],
            self._seq_lengths: batch[1],
            self._weights: batch[2]
        }

        return sess.run(goal, feed_dict=feed_dict)

    @staticmethod
    def load_vocab(path="./quebap/projects/autoread/document.vocab",
                   max_vocab_size=50000):
        vocab = {}
        with open(path, "r") as f:
            for line in f.readlines()[2:max_vocab_size]:
                splits = line.split("\t")
                vocab[splits[1]] = int(splits[0])
        vocab["XXXXX"] = len(vocab)
        return vocab

    @staticmethod
    def vocab_to_ixmap(vocab):
        ixmap = {}
        for word in vocab:
            ixmap[vocab[word]] = word
        return ixmap

    @staticmethod
    def create_from_config(config, devices=None, dropout=0.0, cloze_noise=1.0):
        """
        :param config: dictionary of parameters for creating an autoreader
        :return:
        """

        # todo: load parameters of the model
        # todo: dump config dictionary as json
        autoreader = AutoReader(
            config["size"],
            config["vocab_size"],
            config.get("is_train", True),
            config.get("learning_rate", 1e-2),
            dropout,
            cloze_noise,
            config.get("composition", "GRU"),
            devices,
            config.get("name", "AutoReader"),
            config.get("unk_id", -1),
            config.get("forward_only", False),
        )

        return autoreader
