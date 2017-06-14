# -*- coding: utf-8 -*-

import os
import h5py
import time

import logging
logger = logging.getLogger(__name__)


def write_to_hdf(path, data):
    """
    Writes a numpy array to a hdf5 file under the given path.
    
    :param path: 
    :param data: 
    :return: 
    """
    h5file = h5py.File(path, "w")
    h5file.create_dataset("default", data=data)
    h5file.close()


def load_hdf_file(path, keyword='default'):
    """
    Reads and returns a numpy array for a hdf5 file.
    
    :param path: 
    :param keyword: 
    :return: 
    """
    h5file = h5py.File(path, 'r')
    dset = h5file.get(keyword)
    data = dset[:]
    h5file.close()
    return data


def load_hdf5_paths(paths, limit=None):
    """
    
    :param paths: 
    :param limit: 
    :return: 
    """
    data = []
    for path in paths:
        data.append(load_hdf_file(path)[:limit] if limit else load_hdf_file(path))
    return data


def get_home_path():
    """
    Return the home directory path.
    :return: path.
    """
    return os.environ['HOME']


def get_data_path():
    """
    Return the data path.
    
    :return: path.
    """
    return os.path.join(os.environ['HOME'], '.data')


def make_dirs_if_not_exists(path):
    """
    Creates a directory if it does not exists.
    
    :param path: 
    :return: 
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
