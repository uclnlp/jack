import tensorflow as tf
from jtr.jack import core_models
from jtr.jack.core import SimpleModelModule, Ports

class PairOfBiLSTMOverSupportAndQuestionConditionalEncoding(SimpleModelModule):

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.single_support.name : Ports.Input.single_support,
                Ports.Input.question, Ports.Input.support_length,
                Ports.Input.question_length]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores, Ports.Prediction.candidate_index]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores,
                Ports.Targets.candidate_labels]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    def forward_pass(self, Q, S, Q_lengths, S_lengths, num_candidates):
        # final states_fw_bw dimensions: 
        # [[[batch, output dim], [batch, output_dim]]
        all_states_fw_bw, final_states_fw_bw = core_models.pair_of_bidirectional_LSTMs(
                Q, Q_lengths, S, S_lengths,
                shared_resources.config['out_dim'], drop_keep_prob =
                shared_resources.config['drob_keep_prob'],
                conditional_encoding=True)

        # ->  [batch, 2*output_dim]
        final_states = tf.concat(final_states_fw_bw[0][1],
                                 final_states_fw_bw[1][1],axis=1)

        # [batch, 2*output_dim] -> [batch, num_candidates]
        outputs = core_models.fully_connected_projection(final_states, y, num_candidates)

        return outputs


    def create_output(self, shared_resources: SharedResources,
                      support : tf.Tensor,
                      question : tf.Tensor,
                      support_length : tf.Tensor,
                      question_length : tf.Tensor)
    -> Sequence[tf.Tensor]:
        logits = forward_pass(question, support, question_length,
                support_length, shared_resoures.config['num_candidates'])
        predictions = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')

        return [logits, predictions]


    def create_training_output(self, shared_resources: SharedResources,
                               logits : tf.Tensor,
                               labels : tf.Tensor)
                            -> Sequence[tf.Tensor]:
        predictions = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')
        loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
            labels=labels), name='predictor_loss')
        return [loss]
