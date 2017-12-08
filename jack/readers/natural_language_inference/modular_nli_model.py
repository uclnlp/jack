import tensorflow as tf

from jack.readers.multiple_choice.shared import AbstractSingleSupportMCModel
from jack.readers.natural_language_inference import prediction_layer
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.modular_encoder import modular_encoder


class ModularNLIModel(AbstractSingleSupportMCModel):
    def forward_pass(self, shared_resources, embedded_question, embedded_support, num_classes, tensors):
        [char_emb_question, char_emb_support] = conv_char_embedding(
            len(shared_resources.char_vocab), shared_resources.config['repr_dim'], tensors.word_chars,
            tensors.word_char_length, [tensors.question_words, tensors.support_words])

        model = shared_resources.config['model']
        repr_dim = shared_resources.config['repr_dim']
        input_size = shared_resources.config["repr_dim_input"]
        tensors.emb_question.set_shape([None, None, input_size])
        tensors.emb_support.set_shape([None, None, input_size])

        inputs = {'question': embedded_question, 'support': embedded_support,
                  'char_question': char_emb_question, 'char_support': char_emb_support}
        inputs_length = {'question': tensors.question_length, 'support': tensors.support_length,
                         'char_question': tensors.question_length, 'char_support': tensors.support_length}
        inputs_mapping = {'question': None, 'support': None, 'char_support': None}

        encoder_config = model['encoder_layer']
        encoded, _, _ = modular_encoder(
            encoder_config, inputs, inputs_length, inputs_mapping, repr_dim, tensors.is_eval)

        with tf.variable_scope('prediction_layer'):
            answer_layer_config = model['prediction_layer']
            encoded_question = encoded[answer_layer_config.get('question', 'question')]
            encoded_support = encoded[answer_layer_config.get('support', 'support')]

            if 'repr_dim' not in answer_layer_config:
                answer_layer_config['repr_dim'] = repr_dim
            logits = prediction_layer.prediction_layer(
                encoded_question, tensors.question_length, encoded_support,
                tensors.support_length, num_classes, **answer_layer_config)

        return logits
