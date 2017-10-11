# -*- coding: utf-8 -*-

import logging
import os
import time

logger = logging.getLogger(__name__)


def get_home_path():
    """
    Return the home directory path.

    Returns:
        home directory path
    """
    return os.environ['HOME']


def get_data_path():
    """
    Return the data path.

    Returns:
        data path
    """
    return os.path.join(os.environ['HOME'], '.data')


def make_dirs_if_not_exists(path):
    """
    Creates a directory if it does not exists.

    Args:
        path:

    Returns:

    """
    if not os.path.exists(path):
        os.makedirs(path)


class Timer(object):
    def __init__(self, silent=False):
        self.cumulative_secs = {}
        self.current_ticks = {}
        self.silent = silent

    def tick(self, name='default'):
        if name not in self.current_ticks:
            self.current_ticks[name] = time.time()
            return 0.0
        else:
            if name not in self.cumulative_secs:
                self.cumulative_secs[name] = 0
            t = time.time()
            self.cumulative_secs[name] += t - self.current_ticks[name]
            self.current_ticks.pop(name)

            return self.cumulative_secs[name]

    def tock(self, name='default'):
        self.tick(name)
        value = self.cumulative_secs[name]
        if not self.silent:
            logger.info('Time taken for {0}: {1:.1f}s'.format(name, value))
        self.cumulative_secs.pop(name)
        self.current_ticks.pop(name, None)
        return value
