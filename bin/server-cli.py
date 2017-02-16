#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import configparser

from flask import Flask, request, jsonify
from flask.views import View

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))

app = Flask('jack-the-service')


class InvalidAPIUsage(Exception):
    """
    Class used for handling error messages.
    """
    DEFAULT_STATUS_CODE = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        self.status_code = self.DEFAULT_STATUS_CODE
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidAPIUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


class JackService(View):
    """
    Class handling the requests to the jack-the-service REST Service.
    Each request is processed by the dispatch_request() method.
    """
    methods = ['GET', 'POST']

    def dispatch_request(self):
        """
        This method forwards requests to jack-the-reader.
        The output (a dictionary) is then serialized in JSON, and returned to the client.
        """

        text = request.args.get('text')
        model_name = request.args.get('model')

        if 'text' in request.form:
            text = request.form['text']
        if 'model' in request.form:
            model_name = request.form['model']

        if text is None:
            message = "The service accepts GET and POST requests containing a mandatory 'text' parameter"
            raise InvalidAPIUsage(message, status_code=400)

        models = app.config['MODELS']

        if model_name is None:
            model_name = app.config['DEFAULT_MODEL']

        if model_name not in models:
            message = 'Unknown model: %s' % model_name
            raise InvalidAPIUsage(message, status_code=400)

        # Compute answer
        answer = None

        return jsonify(answer)


class ListAvailableModelsService(View):
    """
    Class handling the 'List Available Models' requests to the REST Service.
    Each request is processed by the dispatch_request() method.
    """
    methods = ['GET', 'POST']

    def dispatch_request(self):
        models = app.config['MODELS']

        default_model = app.config['DEFAULT_MODEL']
        available_models = [model_name for model_name in sorted(models.keys())]

        answer = {}

        if default_model is not None:
            answer['defaultModel'] = default_model

        if available_models is not None:
            answer['availableModels'] = available_models

        return jsonify(answer)


def parse_configuration(configuration_path):
    """
    Read the configuration file of the service.
    Args:
        configuration path (str): Path of the configuration file.
    Returns:
        dict: Dictionary containing the configuration of the service.
    --
    @return dict
    :type configuration_path: str
    :param configuration_path: Path of the configuration file.
    """
    config = configparser.ConfigParser()
    config.read(['service.conf', os.path.expanduser(configuration_path)])

    configuration = {}
    for key, value in config['DEFAULT'].items():
        configuration[key] = value

    if 'port' in configuration:
        configuration['port'] = int(configuration['port'])

    return configuration


def main(argv):
    def formatter_class(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)
    argparser = argparse.ArgumentParser('jack-the-service - REST Service for exposing JTR',
                                        formatter_class=formatter_class)

    argparser.add_argument('--configuration', action='store', type=str, default='~/.jtr/service.conf',
                           help='Path of the configuration file (default: ~/.jtr/service.conf)')
    argparser.add_argument('--bind_address', action='store', type=str, default='127.0.0.1',
                           help='Bind address (default: 127.0.0.1)')
    argparser.add_argument('--port', action='store', type=int, default=5000,
                           help='Port used by the service (default: 5000)')
    argparser.add_argument('--debug', action='store_true',
                           help='Run the service in debug mode')

    args = argparser.parse_args(argv)

    configuration_path = args.configuration
    bind_address = args.bind_address
    port = args.port
    debug = args.debug

    configuration = parse_configuration(configuration_path)

    if 'bind_address' not in configuration:
        configuration['bind_address'] = '127.0.0.1'
    if 'port' not in configuration:
        configuration['port'] = 5000

    if bind_address is not None:
        configuration['bind_address'] = bind_address
    if port is not None:
        configuration['port'] = port

    models_path = configuration['models_path']
    default_model = configuration['default_model']
    models_configuration = configuration['models']

    def join(x, y):
        return os.path.join(x, y)

    def expand(x):
        return os.path.expanduser(x)

    models = {}
    for model_name, model_data in models_configuration.items():
        #architecture_path = expand(join(models_path, model_data['architecture']))
        #tokenizer_path = expand(join(models_path, model_data['tokenizer']))
        #weights_path = expand(join(models_path, model_data['weights']))
        #model = utils.get_model(tokenizer_path, architecture_path, weights_path)

        model = None
        models[model_name] = model

    app.config.update(dict(
        DEFAULT_MODEL=default_model,
        MODELS=models
    ))

    app.add_url_rule('/v1/jack', view_func=JackService.as_view('request'))
    app.add_url_rule('/v1/jack/list', view_func=ListAvailableModelsService.as_view('list'))

    app.run(host=configuration['bind_address'], port=configuration['port'], debug=debug)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])

