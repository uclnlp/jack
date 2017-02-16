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

__author__ = 'pminervini'
__copyright__ = 'UCLMR'


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

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])

