import tensorflow as tf

from jack.core import TensorPortTensors, TensorPort
from jack.readers.extractive_qa.tensorflow.abstract_model import AbstractXQAModelModule
from jack.readers.extractive_qa.tensorflow.answer_layer import answer_layer
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.modular_encoder import modular_encoder


class ModularQAModel(AbstractXQAModelModule):
    def set_beam_size(self, k):
        self._beam_size_assign(k)

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        [char_emb_question, char_emb_support] = conv_char_embedding(
            len(shared_resources.char_vocab), shared_resources.config['repr_dim'], tensors.word_chars,
            tensors.word_char_length, [tensors.question_words, tensors.support_words])

        model = shared_resources.config['model']
        repr_dim = shared_resources.config['repr_dim']
        input_size = shared_resources.config["repr_dim_input"]
        dropout = shared_resources.config.get("dropout")
        tensors.emb_question.set_shape([None, None, input_size])
        tensors.emb_support.set_shape([None, None, input_size])

        inputs = {'question': tensors.emb_question, 'support': tensors.emb_support,
                  'char_question': char_emb_question, 'char_support': char_emb_support,
                  'word_in_question': tf.expand_dims(tensors.word_in_question, 2)}
        inputs_length = {'question': tensors.question_length, 'support': tensors.support_length,
                         'char_question': tensors.question_length, 'char_support': tensors.support_length,
                         'word_in_question': tensors.support_length}
        inputs_mapping = {'question': None, 'support': tensors.support2question,
                          'char_support': tensors.support2question}

        encoder_config = model['encoder_layer']

        encoded, lengths, mappings = modular_encoder(
            encoder_config, inputs, inputs_length, inputs_mapping, repr_dim, dropout, tensors.is_eval)

        with tf.variable_scope('answer_layer'):
            answer_layer_config = model['answer_layer']
            encoded_question = encoded[answer_layer_config.get('question', 'question')]
            encoded_support = encoded[answer_layer_config.get('support', 'support')]

            if 'repr_dim' not in answer_layer_config:
                answer_layer_config['repr_dim'] = repr_dim
            if 'max_span_size' not in answer_layer_config:
                answer_layer_config['max_span_size'] = shared_resources.config.get('max_span_size', 16)
            beam_size = tf.get_variable(
                'beam_size', initializer=shared_resources.config.get('beam_size', 1), dtype=tf.int32, trainable=False)
            beam_size_p = tf.placeholder(tf.int32, [], 'beam_size_setter')
            beam_size_assign = beam_size.assign(beam_size_p)
            self._beam_size_assign = lambda k: self.tf_session.run(beam_size_assign, {beam_size_p: k})

            start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                answer_layer(encoded_question, lengths[answer_layer_config.get('question', 'question')],
                             encoded_support, lengths[answer_layer_config.get('support', 'support')],
                             mappings[answer_layer_config.get('support', 'support')],
                             tensors.answer2support, tensors.is_eval,
                             tensors.correct_start, beam_size=beam_size, **answer_layer_config)

        span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))
