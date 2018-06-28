import copy
import logging

import tensorflow as tf

from jack.util.tf.interaction_layer import interaction_layer
from jack.util.tf.sequence_encoder import encoder

logger = logging.getLogger(__name__)


def _flatten(l):
    if isinstance(l, list):
        return [module for sl in l for module in _flatten(sl)]
    else:
        return [l]


def _unique_module_name(module, layer_depth):
    inp = module.get('input', '')
    inp_str = inp if isinstance(inp, str) else '_'.join(inp)
    name = '_'.join([str(layer_depth), inp_str, module['module']])
    return name


def modular_encoder(encoder_config, inputs, inputs_length, inputs_mapping, default_repr_dim, dropout, is_eval):
    outputs = dict(inputs)
    outputs_length = dict(inputs_length)
    outputs_mapping = dict(inputs_mapping)
    seen_layer = set()

    def encode_module(module):
        module_type = module['module']

        reuse = module['name'] in seen_layer
        seen_layer.add(module['name'])

        if module_type == 'repeat':
            reuse = module.get('reuse')
            for k in range(module['num']):
                prefix = module['name'] + '/' if reuse else '%s_%d/' % (module['name'], k)
                for j, inner_module in enumerate(module['encoder']):
                    # copy this configuration
                    inner_module = copy.deepcopy(inner_module)
                    if 'name' not in inner_module:
                        inner_module['name'] = _unique_module_name(inner_module, j)
                    inner_module['name'] = prefix + inner_module['name']
                    encode_module(inner_module)
            return

        try:
            key = module['input']
            out_key = module.get('output', key)
            if module['module'] in ['concat', 'add', 'mul', 'weighted_add', 'sub']:
                outputs_length[out_key] = outputs_length[key[0]]
                outputs_mapping[out_key] = outputs_mapping.get(key[0])
                if module['module'] == 'concat':
                    outputs[out_key] = tf.concat([outputs[k] for k in key], 2, name=module['name'])
                    return
                if module['module'] == 'add':
                    outputs[out_key] = tf.add_n([outputs[k] for k in key], name=module['name'])
                    return
                if module['module'] == 'sub':
                    outputs[out_key] = tf.subtract(outputs[key[0]], outputs[key[1]], name=module['name'])
                    return
                if module['module'] == 'mul':
                    o = outputs[key[0]]
                    for k in key[1:-1]:
                        o *= outputs[k]
                    outputs[out_key] = tf.multiply(o, outputs[key[-1]], name=module['name'])
                    return
                if module['module'] == 'weighted_add':
                    bias = module.get('bias', 0.0)
                    g = tf.layers.dense(tf.concat([outputs[k] for k in key], 2), outputs[key[0]].get_shape()[-1].value,
                                        tf.sigmoid, bias_initializer=tf.constant_initializer(bias))
                    outputs[out_key] = tf.identity(g * outputs[key[0]] + (1.0 - g) * outputs[key[0]],
                                                   name=module['name'])
                    return
            if 'repr_dim' not in module:
                module['repr_dim'] = default_repr_dim
            if 'dependent' in module:
                dep_key = module['dependent']
                outputs[out_key] = interaction_layer(
                    outputs[key], outputs_length[key],
                    outputs[dep_key], outputs_length[dep_key],
                    outputs_mapping.get(key), outputs_mapping.get(dep_key), reuse=reuse, **module)
            else:
                if module.get('dropout') is True:
                    # set dropout to default dropout
                    module['dropout'] = dropout
                outputs[out_key] = encoder(outputs[key], outputs_length[key], reuse=reuse, is_eval=is_eval, **module)
            outputs_length[out_key] = outputs_length[key]
            outputs_mapping[out_key] = outputs_mapping.get(key)
        except Exception as e:
            logger.error('Creating module %s failed.', module['name'])
            raise e

    encoder_config = _flatten(encoder_config)
    # don't change original config but copy here
    encoder_config = copy.deepcopy(encoder_config)

    for i, module in enumerate(encoder_config):
        if 'name' not in module:
            module['name'] = _unique_module_name(module, i)
        encode_module(module)

    return outputs, outputs_length, outputs_mapping
