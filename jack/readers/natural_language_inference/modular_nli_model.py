import tensorflow as tf

from jack.readers.classification.shared import AbstractSingleSupportClassificationModel
from jack.readers.natural_language_inference import prediction_layer
from jack.util.tf.embedding import conv_char_embedding
from jack.util.tf.modular_encoder import modular_encoder


class ModularNLIModel(AbstractSingleSupportClassificationModel):
    def forward_pass(self, shared_resources, embedded_question, embedded_support, num_classes, tensors):
        model = shared_resources.config['model']
        repr_dim = shared_resources.config['repr_dim']
        dropout = shared_resources.config.get("dropout")

        if shared_resources.config.get('with_char_embeddings'):
            [char_emb_question, char_emb_support] = conv_char_embedding(
                len(shared_resources.char_vocab), shared_resources.config['repr_dim'], tensors.word_chars,
                tensors.word_char_length, [tensors.question_batch_words, tensors.support_batch_words])
            inputs = {'hypothesis': embedded_question, 'premise': embedded_support,
                      'char_hypothesis': char_emb_question, 'char_premise': char_emb_support}
            inputs_length = {'hypothesis': tensors.question_length, 'premise': tensors.support_length,
                             'char_hypothesis': tensors.question_length, 'char_premise': tensors.support_length}
        else:
            inputs = {'hypothesis': embedded_question, 'premise': embedded_support}
            inputs_length = {'hypothesis': tensors.question_length, 'premise': tensors.support_length}

        if dropout:
            for k in inputs:
                inputs[k] = tf.cond(tensors.is_eval, lambda: inputs[k], lambda: tf.nn.dropout(inputs[k], 1.0 - dropout))

        inputs_mapping = {}

        encoder_config = model['encoder_layer']
        encoded, _, _ = modular_encoder(
            encoder_config, inputs, inputs_length, inputs_mapping, repr_dim, dropout, tensors.is_eval)

        with tf.variable_scope('prediction_layer'):
            prediction_layer_config = model['prediction_layer']
            encoded_question = encoded[prediction_layer_config.get('hypothesis', 'hypothesis')]
            encoded_support = encoded[prediction_layer_config.get('premise', 'premise')]

            if 'repr_dim' not in prediction_layer_config:
                prediction_layer_config['repr_dim'] = repr_dim
            prediction_layer_config['dropout'] = dropout if prediction_layer_config.get('dropout', False) else 0.0
            logits = prediction_layer.prediction_layer(
                encoded_question, tensors.question_length, encoded_support,
                tensors.support_length, num_classes, tensors.is_eval, **prediction_layer_config)

        return logits
