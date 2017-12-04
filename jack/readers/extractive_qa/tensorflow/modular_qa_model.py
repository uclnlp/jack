import tensorflow as tf

from jack.core import TensorPortTensors, TensorPort
from jack.readers.extractive_qa.tensorflow.abstract_model import AbstractXQAModelModule
from jack.readers.extractive_qa.tensorflow.answer_layer import answer_layer
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.highway import highway_network
from jack.tfutil.interaction_layer import interaction_layer
from jack.tfutil.sequence_encoder import encoder


class ModularQAModel(AbstractXQAModelModule):
    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)

        encoded_question, encoded_support = self.embedding_layer(shared_resources, tensors)
        keep_prob = 1.0 - shared_resources.config.get("dropout", 0.0)
        encoded_question, encoded_support = tf.cond(
            tensors.is_eval,
            lambda: (encoded_question, encoded_support),
            lambda: (
                tf.nn.dropout(encoded_question, keep_prob, noise_shape=[1, 1, encoded_question.get_shape()[-1].value]),
                tf.nn.dropout(encoded_support, keep_prob, noise_shape=[1, 1, encoded_support.get_shape()[-1].value]))
        )

        model = shared_resources.config['model']
        repr_dim = shared_resources.config['repr_dim']

        inputs = {'question': encoded_question, 'support': encoded_support}
        inputs_length = {'question': tensors.question_length, 'support': tensors.support_length}
        inputs_mapping = {'question': None, 'support': tensors.support2question}

        for i, layer in enumerate(model['encoder_layer']):
            key = layer['input']
            out_key = layer.get('output', key)
            if layer['layer_type'] == 'word_in_question':
                key = layer['input']
                # add 2x wiq to keep dimensionality even, otherwise FusedLSTM complains...
                inputs[out_key] = tf.concat([inputs[key],
                                             tf.stack([tensors.word_in_question, tensors.word_in_question], 2)], 2)
                continue

            if layer['layer_type'] == 'concat':
                inputs[out_key] = tf.concat([inputs[k] for k in key], 2)
                continue
            if 'name' not in layer:
                layer['name'] = '_'.join([str(i), layer['input'], layer['layer_type']])
            if 'repr_dim' not in layer:
                layer['repr_dim'] = repr_dim
            if 'dependent' in layer:
                dep_key = layer['dependent']
                inputs[out_key] = interaction_layer(
                    inputs[key], inputs_length[key],
                    inputs[dep_key], inputs_length[dep_key],
                    inputs_mapping[key], **layer)
            else:
                inputs[out_key] = encoder(inputs[key], inputs_length[key], **layer)
            inputs_length[out_key] = inputs_length[key]
            inputs_mapping[out_key] = inputs_mapping[key]
            dropout = layer.get('dropout', shared_resources.config.get("dropout", 0.0))
            if dropout > 0.0:
                inputs[out_key] = tf.cond(
                    tensors.is_eval,
                    lambda: inputs[out_key],
                    lambda: tf.nn.dropout(
                        inputs[out_key], 1.0 - dropout, noise_shape=[1, 1, inputs[out_key].get_shape()[-1].value]))

        with tf.variable_scope('answer_layer'):
            answer_layer_config = model['answer_layer']
            encoded_question = inputs[answer_layer_config.get('question', 'question')]
            encoded_support = inputs[answer_layer_config.get('support', 'support')]

            if 'repr_dim' not in answer_layer_config:
                answer_layer_config['repr_dim'] = repr_dim
            start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                answer_layer(encoded_question, tensors.question_length, encoded_support,
                             tensors.support_length,
                             tensors.support2question, tensors.answer2support, tensors.is_eval,
                             tensors.correct_start, **answer_layer_config)

        span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

        return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span))

    def embedding_layer(self, shared_resources, tensors):
        with tf.variable_scope('embedding_layer') as vs:
            # embedding layer
            input_size = shared_resources.config["repr_dim_input"]
            size = shared_resources.config["repr_dim"]
            with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)
            # 1. char embeddings + word embeddings
            # set shapes for inputs
            tensors.emb_question.set_shape([None, None, input_size])
            tensors.emb_support.set_shape([None, None, input_size])
            emb_question = tensors.emb_question
            emb_support = tensors.emb_support
            # 1. + 2a. + 2b. 2a. char embeddings + conv + max pooling
            if with_char_embeddings:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    len(shared_resources.char_vocab), size, tensors.word_chars, tensors.word_char_length,
                    [tensors.question_words, tensors.support_words])
                # 3. cat
                emb_question = tf.concat([emb_question, char_emb_question], 2)
                emb_support = tf.concat([emb_support, char_emb_support], 2)
                # apply projection
                emb_question = tf.layers.dense(emb_question, size, tf.tanh, name='projection')
                # highway layer to allow for interaction between concatenated embeddings
                # 3. highway
                emb_question = highway_network(emb_question, 1, scope='embedding_highway')
                vs.reuse_variables()
                emb_support = tf.layers.dense(emb_support, size, tf.tanh, name='projection')
                emb_support = highway_network(emb_support, 1, scope='embedding_highway')
            else:
                # apply projection
                emb_support = tf.layers.dense(emb_support, size, tf.tanh, name='projection')
                vs.reuse_variables()
                emb_question = tf.layers.dense(emb_question, size, tf.tanh, name='projection')

            return emb_question, emb_support
