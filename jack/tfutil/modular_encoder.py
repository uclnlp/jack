import logging

import tensorflow as tf

from jack.tfutil.interaction_layer import interaction_layer
from jack.tfutil.sequence_encoder import encoder

logger = logging.getLogger(__name__)


def modular_encoder(encoder_config, inputs, inputs_length, inputs_mapping, default_repr_dim, dropout, is_eval):
    outputs = dict(inputs)
    outputs_length = dict(inputs_length)
    outputs_mapping = dict(inputs_mapping)
    seen_layer = set()
    for i, module in enumerate(encoder_config):
        if 'name' not in module:
            inp_str = module['input'] if isinstance(module['input'], str) else '_'.join(module['input'])
            module['name'] = '_'.join([str(i), inp_str, module['module']])
        reuse = module['name'] in seen_layer
        seen_layer.add(module['name'])
        try:
            key = module['input']
            out_key = module.get('output', key)
            if module['module'] in ['concat', 'add', 'mul', 'weighted_add', 'sub']:
                outputs_length[out_key] = outputs_length[key[0]]
                outputs_mapping[out_key] = outputs_mapping.get(key[0])
                if module['module'] == 'concat':
                    outputs[out_key] = tf.concat([outputs[k] for k in key], 2, name=module['name'])
                    continue
                if module['module'] == 'add':
                    outputs[out_key] = tf.add_n([outputs[k] for k in key], name=module['name'])
                    continue
                if module['module'] == 'sub':
                    outputs[out_key] = tf.subtract(outputs[key[0]], outputs[key[1]], name=module['name'])
                    continue
                if module['module'] == 'mul':
                    o = outputs[key[0]]
                    for k in key[1:-1]:
                        o *= outputs[k]
                    outputs[out_key] = tf.multiply(o, outputs[key[-1]], name=module['name'])
                    continue
                if module['module'] == 'weighted_add':
                    bias = module.get('bias', 0.0)
                    g = tf.layers.dense(tf.concat([outputs[k] for k in key], 2), outputs[key[0]].get_shape()[-1].value,
                                        tf.sigmoid, bias_initializer=tf.constant_initializer(bias))
                    outputs[out_key] = tf.identity(g * outputs[key[0]] + (1.0 - g) * outputs[key[0]],
                                                   name=module['name'])
                    continue

            if isinstance(key, list):
                # auto-concat
                new_key = '_'.join(key)
                outputs[new_key] = tf.concat([outputs[k] for k in key], 2)
                key = new_key

            if 'repr_dim' not in module:
                module['repr_dim'] = default_repr_dim
            if 'dependent' in module:
                dep_key = module['dependent']
                outputs[out_key] = interaction_layer(
                    outputs[key], outputs_length[key],
                    outputs[dep_key], outputs_length[dep_key],
                    outputs_mapping.get(key), outputs_mapping.get(dep_key), reuse=reuse, **module)
            else:
                outputs[out_key] = encoder(outputs[key], outputs_length[key], reuse=reuse, **module)
            outputs_length[out_key] = outputs_length[key]
            outputs_mapping[out_key] = outputs_mapping.get(key)
            if module.get('dropout', False):
                outputs[out_key] = tf.cond(
                    is_eval,
                    lambda: outputs[out_key],
                    lambda: tf.nn.dropout(
                        outputs[out_key], 1.0 - dropout, noise_shape=[1, 1, outputs[out_key].get_shape()[-1].value]))
        except Exception as e:
            logger.error('Creating module %s failed.', module['name'])
            raise e
    return outputs, outputs_length, outputs_mapping
