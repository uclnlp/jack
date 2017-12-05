import logging

import tensorflow as tf

from jack.tfutil.interaction_layer import interaction_layer
from jack.tfutil.sequence_encoder import encoder

logger = logging.getLogger(__name__)


def modular_encoder(encoder_config, inputs, inputs_length, inputs_mapping, default_repr_dim, is_eval):
    outputs = dict(inputs)
    outputs_length = dict(inputs_length)
    outputs_mapping = dict(inputs_mapping)
    seen_layer = set()
    for i, layer in enumerate(encoder_config):
        if 'name' not in layer:
            inp_str = layer['input'] if isinstance(layer['input'], str) else '_'.join(layer['input'])
            layer['name'] = '_'.join([str(i), inp_str, layer['module']])
        reuse = layer['name'] in seen_layer
        seen_layer.add(layer['name'])
        try:
            key = layer['input']
            out_key = layer.get('output', key)
            if layer['module'] == 'concat':
                outputs[out_key] = tf.concat([outputs[k] for k in key], 2)
                continue
            if layer['module'] == 'add':
                outputs[out_key] = tf.add_n([outputs[k] for k in key], 2)
                continue
            if layer['module'] == 'mul':
                outputs[out_key] = outputs[key[0]]
                for k in key[1:]:
                    outputs[out_key] *= outputs[k[0]]
                continue
            if layer['module'] == 'weighted_add':
                bias = layer.get('bias', 0.0)
                g = tf.layers.dense(tf.concat([outputs[k] for k in key], 2), outputs[key[0]].get_shape()[-1].value,
                                    tf.sigmoid, bias_initializer=tf.constant_initializer(bias))
                outputs[out_key] = g * outputs[key[0]] + (1.0 - g) * outputs[key[0]]
                continue
            if 'repr_dim' not in layer:
                layer['repr_dim'] = default_repr_dim
            if 'dependent' in layer:
                dep_key = layer['dependent']
                outputs[out_key] = interaction_layer(
                    outputs[key], outputs_length[key],
                    outputs[dep_key], outputs_length[dep_key],
                    outputs_mapping[key], reuse=reuse, **layer)
            else:
                outputs[out_key] = encoder(outputs[key], outputs_length[key], reuse=reuse, **layer)
            outputs_length[out_key] = outputs_length[key]
            outputs_mapping[out_key] = outputs_mapping[key]
            dropout = layer.get('dropout', 0.0)
            if dropout > 0.0:
                outputs[out_key] = tf.cond(
                    is_eval,
                    lambda: outputs[out_key],
                    lambda: tf.nn.dropout(
                        outputs[out_key], 1.0 - dropout, noise_shape=[1, 1, outputs[out_key].get_shape()[-1].value]))
        except Exception as e:
            logger.error('Creating module %s failed.', layer['name'])
            raise e
    return outputs, outputs_length, outputs_mapping
