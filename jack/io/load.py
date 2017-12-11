"""Implementation of loaders for common datasets."""

import json

from jack.core.data_structures import *
from jack.io.SNLI2jtr import convert_snli
from jack.io.SQuAD2jtr import convert_squad

loaders = dict()


def _register(name):
    def _decorator(f):
        loaders[name] = f
        return f

    return _decorator


@_register('jack')
def load_jack(path, max_count=None):
    """
    This function loads a jack json file from a specific location.
    Args:
        path: the location to load from.
        max_count: how many instances to load at most

    Returns:
        A list of input-answer pairs.

    """
    # We load json directly instead
    with open(path) as f:
        jtr_data = json.load(f)

    return jack_to_qasetting(jtr_data, max_count)


@_register('squad')
def load_squad(path, max_count=None):
    """
    This function loads a squad json file from a specific location.
    Args:
        path: the location to load from.
        max_count: how many instances to load at most

    Returns:
        A list of input-answer pairs.
    """
    # We load to jtr dict and convert to qa settings for now
    jtr_data = convert_squad(path)
    return jack_to_qasetting(jtr_data, max_count)


@_register('snli')
def load_snli(path, max_count=None):
    """
    This function loads a jack json file with labelled answers from a specific location.
    Args:
        path: the location to load from.
        max_count: how many instances to load at most

    Returns:
        A list of input-answer pairs.
    """
    # We load to jtr dict and convert to qa settings for now
    jtr_data = convert_snli(path)
    return jack_to_qasetting(jtr_data, max_count)
