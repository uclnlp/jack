import tensorflow as tf

from jack.readers.classification.shared import AbstractSingleSupportClassificationModel
from jack.readers.natural_language_inference import prediction_layer
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.modular_encoder import modular_encoder


class ModularNLIModel(AbstractSingleSupportClassificationModel):
    def forward_pass(self, shared_resources, embedded_question, embedded_support, num_classes, tensors):
        [char_emb_question, char_emb_support] = conv_char_embedding(
            len(shared_resources.char_vocab), shared_resources.config['repr_dim'], tensors.word_chars,
            tensors.word_char_length, [tensors.question_words, tensors.support_words])

        model = shared_resources.config['model']
        repr_dim = shared_resources.config['repr_dim']
        input_size = shared_resources.config["repr_dim_input"]
        dropout = shared_resources.config.get("dropout")
        tensors.emb_question.set_shape([None, None, input_size])
        tensors.emb_support.set_shape([None, None, input_size])

        inputs = {'hypothesis': embedded_question, 'premise': embedded_support,
                  'char_hypothesis': char_emb_question, 'char_premise': char_emb_support}
        inputs_length = {'hypothesis': tensors.question_length, 'premise': tensors.support_length,
                         'char_hypothesis': tensors.question_length, 'char_premise': tensors.support_length}
        inputs_mapping = {'hypothesis': None, 'premise': None, 'char_premise': None, 'char_hypothesis': None}

        encoder_config = model['encoder_layer']
        encoded, _, _ = modular_encoder(
            encoder_config, inputs, inputs_length, inputs_mapping, repr_dim, dropout, tensors.is_eval)

        with tf.variable_scope('prediction_layer'):
            answer_layer_config = model['prediction_layer']
            encoded_question = encoded[answer_layer_config.get('hypothesis', 'hypothesis')]
            encoded_support = encoded[answer_layer_config.get('premise', 'premise')]

            if 'repr_dim' not in answer_layer_config:
                answer_layer_config['repr_dim'] = repr_dim
            logits = prediction_layer.prediction_layer(
                encoded_question, tensors.question_length, encoded_support,
                tensors.support_length, num_classes, **answer_layer_config)

        return logits
