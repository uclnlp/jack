import tensorflow as tf
from jtr.jack import core_models

class PairOfBiLSTMOverSupportAndQuestionConditionalEncoding(SimpleModelModule):


    def forward_pass(self, shared_resources, input_tensors):
        Q = input_tensors['question']
        S = input_tensors['single_support']
        S_lengths = input_tensors['support_length_flat']
        Q_lengths = input_tensors['question_length_flat']
        y = input_tensors['candidates']
        num_candidates = shared_resources.config['num_candidates']

        # final states_fw_bw dimensions: 
        # [[batch, output dim], [batch, output_dim]]
        all_states_fw_bw, final_states_fw_bw = core_models.pair_of_bidirectional_LSTMs(
                Q, Q_lengths, S, S_lengths,
                shared_resources.config['out_dim'], drop_keep_prob =
                shared_resources.config['drob_keep_prob'],
                conditional_encoding=True)

        # ->  [batch, 2*output_dim]
        final_states = tf.concat(final_states_fw_bw, axis=1)

        # [batch, 2*output_dim] -> [batch, num_candidates]
        outputs = core_models.fully_connected_projection(final_states, y, num_candidates)

        return outputs


    def create_output(self, shared_resources: SharedResources,
                      input_tensors : Dict)
    -> Dict[String : tf.Tensor]:
        """
        This function needs to be implemented in order to define how the module produces
        output from input tensors corresponding to `input_ports`.
        Args:
            *input_tensors: a list of input tensors.

        Returns:
            mapping from output ports to their tensors.
        """
        logits = forward_pass(shared_resources, input_tensors)
        predictions = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')

        out_port = Ports.Prediction.candidate_idx


        return {out_port.name : out_port}


    def create_training_output(self, shared_resources: SharedResources,
                               *training_input_tensors: tf.Tensor)
                            -> Sequence[tf.Tensor]:
        """
        This function needs to be implemented in order to define how the module produces tensors only used
        during training given tensors corresponding to the ones defined by `training_input_ports`, which might include
        tensors corresponding to ports defined by `output_ports`. This sub-graph should only be created during training.
        Args:
            *training_input_tensors: a list of input tensors.

        Returns:
            mapping from output ports to their tensors.
        """
        logits = forward_pass(shared_resources, input_tensors)
        predictions = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')
        loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets), name='predictor_loss')
        loss, predictions
