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
    dir = path + "/" + current_time + "/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    if name is not None:
        if os.path.exists(path + "/" + name):
            os.remove(path + "/" + name)
        os.symlink(current_time, path + "/" + name, target_is_directory=True)
    if link_to_latest:
        if os.path.exists(path + "/latest"):
            os.remove(path + "/latest")
        os.symlink(current_time, path + "/latest", target_is_directory=True)
    return dir


def save_conf(path, conf):
    with open(path, "w") as f_out:
        splits = path.split("/")
        dir = "/".join(splits[:-1]) + "/"
        conf["meta"]["experiment_dir"] = dir
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