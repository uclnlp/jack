# -*- coding: utf-8 -*-

#         __  _ __
#  __  __/ /_(_) /
# / / / / __/ / /
#/ /_/ / /_/ / /
#\____/\__/_/_/ v0.2
#
#Making useful stuff happen since 2016

import numpy as np
import contextlib
from time import gmtime, strftime
import os
import json
import tensorflow as tf
import h5py
import time

import logging
logger = logging.getLogger("tfutil")


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Switches printoptions temporarily via yield before switching back."""
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def shape2str(x):
    """Converts a shape array to a string."""
    return "[" + " x ".join([str(x) for x in x.shape]) + "]"


def nprint(x, prefix="", precision=3, surpress=True, max_list_len=5, show_shape=True):
    """Prints `x` with given numpy options (default=compact+shape)."""
    with printoptions(precision=precision, suppress=surpress):
        if isinstance(x, np.ndarray):
            print(prefix + "ndarray")
            print(str(x))
            if show_shape:
                print("of shape " + shape2str(x) + "\n")
        elif isinstance(x, tuple):
            print(prefix + "tuple")
            for i, j in enumerate(x):
                print(str(i))
                nprint(j, prefix, precision, surpress, max_list_len, show_shape)
            print()
        elif isinstance(x, list):
            # fixme: breaks when list elements are not ndarrays
            print(prefix + "list of %d elements with shape %s"
                  % (len(x), shape2str(x[0])))
            for i in range(min(len(x), max_list_len)):
                nprint(x[i], prefix + "list[%d] " % i, precision, surpress, max_list_len, show_shape=False)
            print()
        # todo: do the same for tensors
        else:
            print(x)
        print()


def tfprint(tensor, message="", precision=2, first_n=None, summarize=10000,
            name=None, print_shape=True):
    def print_tensor(x):
        str_val = message
        str_val += np.array2string(x, precision=precision)
        if print_shape:
            str_val += "\n" + str(x.shape)
        logger.debug(str_val)
        return x

    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)
        return res


def tfprint_legacy(tensor, message=None, precision=5, first_n=None, summarize=10000,
            name=None, print_shape=True):
    def reduce_precision(a, precision=2):
        return tf.floordiv(a * 100, 1) / 100
    tmp = tf.Print(tensor, [reduce_precision(tensor, precision=precision)],
                   message=message, first_n=first_n,
                   summarize=summarize, name=name)
    if print_shape:
        tmp = tf.Print(tmp, [tf.shape(tmp)], message="shape_" + message)
    return tmp


def get_timestamped_dir(path, name=None, link_to_latest=False):
    """Create a directory with the current timestamp."""
    current_time = strftime("%y-%m-%d/%H-%M-%S", gmtime())
    folder_path = path + "/" + current_time + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if name is not None:
        if os.path.exists(path + "/" + name):
            os.remove(path + "/" + name)
        os.symlink(current_time, path + "/" + name, target_is_directory=True)
    if link_to_latest:
        if os.path.exists(path + "/latest"):
            os.remove(path + "/latest")
        os.symlink(current_time, path + "/latest", target_is_directory=True)
    return folder_path


def save_conf(path, conf):
    with open(path, "w") as f_out:
        splits = path.split("/")
        folder_path = "/".join(splits[:-1]) + "/"
        conf["meta"]["experiment_dir"] = folder_path
        json.dump(conf, f_out, indent=4, sort_keys=True)
        f_out.close()


def deep_merge(dict1, dict2):
    """
    overrides entries in dict1 with entries in dict2!
    """
    if isinstance(dict1, dict) and isinstance(dict2, dict):
        tmp = {}
        for key in dict1:
            if key not in dict2:
                tmp[key] = dict1[key]
            else:
                tmp[key] = deep_merge(dict1[key], dict2[key])
        for key in dict2:
            if key not in dict1:
                tmp[key] = dict2[key]
        return tmp
    else:
        return dict2


def load_conf(path, experiment_dir=None):
    file_name = path.split("/")[-1]

    with open(path, 'r') as f:
        conf = eval(f.read())

        if "meta" not in conf:
            conf["meta"] = {}

        conf["meta"]["conf"] = path
        conf["meta"]["name"] = file_name.split(".")[0]
        conf["meta"]["file_name"] = file_name

        if "parent" in conf["meta"] and conf["meta"]["parent"] is not None:
            parent = load_conf(conf["meta"]["parent"])
            conf = deep_merge(parent, conf)  # {**parent, **conf}

        if experiment_dir is not None:
            save_conf(experiment_dir+file_name, conf)

        f.close()

        return conf


def write_to_hdf(path, data):
    '''Writes a numpy array to a hdf5 file under the given path.'''
    h5file = h5py.File(path, "w")
    h5file.create_dataset("default", data=data)
    h5file.close()


def load_hdf_file(path, keyword='default'):
    '''Reads and returns a numpy array for a hdf5 file'''
    h5file = h5py.File(path, 'r')
    dset = h5file.get(keyword)
    data = dset[:]
    h5file.close()
    return data

def load_hdf5_paths(paths, limit=None):
    data = []
    for path in paths:
        if limit != None:
            data.append(load_hdf_file(path)[:limit])
        else:
            data.append(load_hdf_file(path))
    return data

def get_home_path():
    return os.environ['HOME']

def get_data_path():
    return os.path.join(os.environ['HOME'], '.data')

def make_dirs_if_not_exists(path):
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
            print('Time taken for {0}: {1:.1f}s'.format(name, value))
        self.cumulative_secs.pop(name)
        self.current_ticks.pop(name, None)

        return value


