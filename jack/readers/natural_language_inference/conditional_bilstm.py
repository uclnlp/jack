import tensorflow as tf

from jack.readers.classification.shared import AbstractSingleSupportClassificationModel
from jack.util.tf.rnn import fused_birnn


class ConditionalBiLSTMClassificationModel(AbstractSingleSupportClassificationModel):
    def forward_pass(self, shared_resources, embedded_question, embedded_support, num_classes, tensors):
        # question - hypothesis; support - premise
        repr_dim = shared_resources.config['repr_dim']
        dropout = shared_resources.config.get("dropout", 0.0)

        with tf.variable_scope('embedding_projection') as vs:
            embedded_question = tf.layers.dense(embedded_question, repr_dim, tf.tanh, name='projection')
            vs.reuse_variables()
            embedded_support = tf.layers.dense(embedded_support, repr_dim, tf.tanh, name='projection')
            # keep dropout mask constant over time
            dropout_shape = [tf.shape(embedded_question)[0], 1, tf.shape(embedded_question)[2]]
            embedded_question = tf.nn.dropout(embedded_question, 1.0 - dropout, dropout_shape)
            embedded_support = tf.nn.dropout(embedded_support, 1.0 - dropout, dropout_shape)

        fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(repr_dim)
        # [batch, 2*output_dim] -> [batch, num_classes]
        _, q_states = fused_birnn(fused_rnn, embedded_question, sequence_length=tensors.question_length,
                                  dtype=tf.float32, time_major=False, scope="question_rnn")

        outputs, _ = fused_birnn(fused_rnn, embedded_support, sequence_length=tensors.support_length,
                                 dtype=tf.float32, initial_state=q_states, time_major=False, scope="support_rnn")

        # [batch, T, 2 * dim] -> [batch, dim]
        outputs = tf.concat([outputs[0], outputs[1]], axis=2)
        hidden = tf.layers.dense(outputs, repr_dim, tf.nn.relu, name="hidden") * tf.expand_dims(
            tf.sequence_mask(tensors.support_length, maxlen=tf.shape(outputs)[1], dtype=tf.float32), 2)
        hidden = tf.reduce_max(hidden, axis=1)
        # [batch, dim] -> [batch, num_classes]
        outputs = tf.layers.dense(hidden, num_classes, name="classification")
        return outputs
