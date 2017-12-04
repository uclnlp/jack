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

        with tf.variable_scope('contextual_encoder'):
            for i, layer in enumerate(model['question_layer']):
                if 'name' not in layer:
                    layer['name'] = 'question_%d' % i
                if 'repr_dim' not in layer:
                    layer['repr_dim'] = repr_dim
                encoded_question = encoder(encoded_question, tensors.question_length, **layer)
            for i, layer in enumerate(model['support_layer']):
                if 'name' not in layer:
                    layer['name'] = 'support_%d' % i
                if 'repr_dim' not in layer:
                    layer['repr_dim'] = repr_dim
                encoded_support = encoder(encoded_support, tensors.support_length, **layer)

        with tf.variable_scope('interaction'):
            interaction_layer_config = model.get('interaction_layer')
            if interaction_layer_config:
                if 'repr_dim' not in interaction_layer_config:
                    interaction_layer_config['repr_dim'] = repr_dim
                encoded_support = interaction_layer(
                    encoded_support, tensors.support_length, encoded_question, tensors.question_length,
                    tensors.support2question, layer_type=interaction_layer_config['layer_type'],
                    attn_type=interaction_layer_config.get('attn_type', 'diagonal_bilinear'),
                    scaled=interaction_layer_config.get('scaled', True),
                    name=interaction_layer_config.get('name', 'interaction_attention'),
                )
        if shared_resources.config.get('word_in_question', False):
            # add 2x wiq to keep dimensionality even, otherwise FusedLSTM complains...
            encoded_support = tf.concat([encoded_support,
                                         tf.stack([tensors.word_in_question, tensors.word_in_question], 2)], 2)

        with tf.variable_scope('encoder'):
            for i, layer in enumerate(model.get('encoder_layer', [])):
                if 'name' not in layer:
                    layer['name'] = 'encoder_%d' % i
                if 'repr_dim' not in layer:
                    layer['repr_dim'] = repr_dim
                encoded_support = encoder(encoded_support, tensors.support_length, **layer)

        encoded_question, encoded_support = tf.cond(
            tensors.is_eval,
            lambda: (encoded_question, encoded_support),
            lambda: (
            tf.nn.dropout(encoded_question, keep_prob, noise_shape=[1, 1, encoded_question.get_shape()[-1].value]),
            tf.nn.dropout(encoded_support, keep_prob, noise_shape=[1, 1, encoded_support.get_shape()[-1].value]))
        )

        with tf.variable_scope('answer_layer'):
            answer_layer_config = model['answer_layer']
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
