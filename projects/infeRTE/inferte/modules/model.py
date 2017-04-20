# -*- coding: utf-8 -*-

import tensorflow as tf

from jtr.jack.tf_fun import rnn, simple
from jtr.jack.tasks.mcqa.abstract_multiplechoice import AbstractSingleSupportFixedClassModel


class PairOfBiLSTMOverSupportAndQuestionModel(AbstractSingleSupportFixedClassModel):
    def forward_pass(self, shared_resources, nvocab,
                     Q, S, Q_lengths, S_lengths,
                     num_classes):
        # final states_fw_bw dimensions:
        # [[[batch, output dim], [batch, output_dim]]

        Q_seq = nvocab(Q)
        S_seq = nvocab(S)

        all_states_fw_bw, final_states_fw_bw = rnn.pair_of_bidirectional_LSTMs(
                Q_seq, Q_lengths, S_seq, S_lengths,
                shared_resources.config['repr_dim'], drop_keep_prob =
                1.0-shared_resources.config['dropout'],
                conditional_encoding=True)

        # ->  [batch, 2*output_dim]
        final_states = tf.concat([final_states_fw_bw[0][1],
                                 final_states_fw_bw[1][1]],axis=1)

        # [batch, 2*output_dim] -> [batch, num_classes]
        outputs = simple.fully_connected_projection(final_states,
                                                         num_classes)

        return outputs
