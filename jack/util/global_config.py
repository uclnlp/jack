# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)

class Backends:
    TENSORFLOW = 'tensorflow'
    TEST = 'test'


class Config:
    dropout = 0.0
    batch_size = 128
    learning_rate = 0.001
    backend = Backends.TENSORFLOW
    L2 = 0.000
    cuda = False
    embedding_dim = 128
    hidden_size = 256

    @staticmethod
    def parse_argv(argv):
        args = argv[1:]
        assert len(args) % 2 == 0, 'Global parser expects an even number of arguments.'
        values = []
        names = []
        for i, token in enumerate(args):
            if i % 2 == 0:
                names.append(token)
            else:
                values.append(token)

        for i in range(len(names)):
            if names[i] in alias2params:
                logger.debug('Replaced parameters alias {0} with name {1}', names[i], alias2params[names[i]])
                names[i] = alias2params[names[i]]

        for i in range(len(names)):
            name = names[i]
            if name[:2] == '--':
                continue
            assert name in params2type, 'Parameter {0} does not exist. Prefix your custom parameters with -- ' \
                                        'to skip parsing for global config'.format(name)
            values[i] = params2type[name](values[i])

        for name, value in zip(names, values):
            print(name, value)
            if name[:2] == '--':
                continue
            params2field[name](value)
            logger.debug('Set parameter {0} to {1}', name, value)


params2type = {
    'learning_rate': lambda x: float(x),
    'dropout': lambda x: float(x),
    'batch_size': lambda x: int(x),
    'L2': lambda x: float(x),
    'embedding_dim': lambda x: int(x),
    'hidden_size': lambda x: int(x)
}

alias2params = {
    'lr': 'learning_rate',
    'l2': 'L2'
}

params2field = {
    'learning_rate': lambda x: setattr(Config, 'learning_rate', x),
    'dropout': lambda x: setattr(Config, 'dropout', x),
    'batch_size': lambda x: setattr(Config, 'batch_size', x),
    'L2': lambda x: setattr(Config, 'L2', x),
    'embedding_dim': lambda x: setattr(Config, 'embedding_dim', x),
    'hidden_size': lambda x: setattr(Config, 'embedding_dim', x)
}
