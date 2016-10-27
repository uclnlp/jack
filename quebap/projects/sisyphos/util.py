"""
         __  _ __
  __  __/ /_(_) /
 / / / / __/ / /
/ /_/ / /_/ / /
\__,_/\__/_/_/ v0.2

Making useful stuff happen since 2016
"""

import numpy as np
import contextlib
from time import gmtime, strftime
import os

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def shape2str(x):
    return "[" + " x ".join([str(x) for x in x.shape]) + "]"


def nprint(x, prefix="", precision=3, surpress=True, max_list_len=5, show_shape=True):
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


def get_timestamped_dir(path):
    current_time = strftime("%y-%m-%d/%H-%M-%S", gmtime())
    dir = path + "/" + current_time + "/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
