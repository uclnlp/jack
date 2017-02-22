import tensorflow as tf
from jtr.jack import core_models
from jtr.jack.core import *
from typing import List
from jtr.preprocess.vocab import NeuralVocab

class PairOfBiLSTMOverSupportAndQuestionConditionalEncoding(SimpleModelModule):
    def __init__(self, shared_resources):
        self.nvocab = None
        self.shared_resources = shared_resources

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.single_support,
                Ports.Input.question, FlatPorts.Input.support_length,
                FlatPorts.Input.question_length, FlatPorts.Target.candidate_idx]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores, FlatPorts.Prediction.candidate_idx,
                FlatPorts.Target.candidate_idx]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores,
                FlatPorts.Target.candidate_idx]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    def forward_pass(self, shared_resources, Q, S, Q_lengths, S_lengths, num_candidates):
        # final states_fw_bw dimensions: 
        # [[[batch, output dim], [batch, output_dim]]
        if self.nvocab == None:
            self.nvocab = NeuralVocab(shared_resources.vocab,
                    input_size=shared_resources.config['repr_dim_input'])

        Q_seq = self.nvocab(Q)
        S_seq = self.nvocab(S)

        all_states_fw_bw, final_states_fw_bw = core_models.pair_of_bidirectional_LSTMs(
                Q_seq, Q_lengths, S_seq, S_lengths,
                shared_resources.config['repr_dim'], drop_keep_prob =
                1.0-shared_resources.config['dropout'],
                conditional_encoding=True)

        # ->  [batch, 2*output_dim]
        final_states = tf.concat([final_states_fw_bw[0][1],
                                 final_states_fw_bw[1][1]],axis=1)

        # [batch, 2*output_dim] -> [batch, num_candidates]
        outputs = core_models.fully_connected_projection(final_states, num_candidates)
        print('outputs', outputs.get_shape())

        return outputs


    def create_output(self, shared_resources: SharedResources,
                      support : tf.Tensor,
                      question : tf.Tensor,
                      support_length : tf.Tensor,
                      question_length : tf.Tensor,
                      labels : tf.Tensor) -> Sequence[tf.Tensor]:

        print(shared_resources.config['answer_size'])
        logits = self.forward_pass(shared_resources, question, support, question_length,
                support_length, shared_resources.config['answer_size'])
        predictions = tf.arg_max(logits, 1, name='prediction')

        return [logits, predictions, labels]


    def create_training_output(self, shared_resources: SharedResources,
                               logits : tf.Tensor,
                               labels : tf.Tensor) -> Sequence[tf.Tensor]:

        print(labels.get_shape())
        loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
            labels=labels), name='predictor_loss')
        return [loss]
