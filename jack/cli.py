# -*- coding: utf-8 -*-

import argparse

from jack.readers.multiple_choice.simple_mcqa import *

from jack.util.vocab import Vocab


def example_reader(config):
    resources = SharedResources(Vocab(), config)
    input_module = SimpleMCInputModule(resources)
    model_module = SimpleMCModelModule(resources)
    output_module = SimpleMCOutputModule()
    reader = JTReader(resources,input_module, model_module, output_module)
    return reader


def main():
    # get configuration (from command line, and/or config file)
    config = get_config()
    reader = globals()[config.reader](config)
    if config.predict:
        input_data = None
        # todo: does the reader do internal batching?
        output_data = reader(input_data)
        # store output
    else:
        train_data = None
        dev_data = None
        test_data = None
        # todo: train should accept train and dev and maybe test data
        reader.train(train_data, dev_data, test_data, train_params=config)
        reader.store()


def get_config():
    parser = argparse.ArgumentParser(description='Train and Evaluate a Machine Reader')
    parser.add_argument('--reader', default='example_reader', choices=["example_reader"], help="Reading model to use")
    parser.add_argument('--predict', default=True, type=bool, help='Prediction/Test mode')
    parser.add_argument('--train', default=None, type=argparse.FileType('r'), help="jack training file")
    parser.add_argument('--dev', default=None, type=argparse.FileType('r'), help="jack dev file")
    parser.add_argument('--test', default=None, type=argparse.FileType('r'), help="jack test file")
    parser.add_argument('--out', default=None, type=argparse.FileType('r'), help="jack output file")
    config = parser.parse_args()
    return config
