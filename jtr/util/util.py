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

def load_conf(path, experiment_dir=None):
    file_name = path.split("/")[-1]

    with open(path, 'r') as f:
        conf = eval(f.read())

        if "meta" not in conf:
            conf["meta"] = {}

        conf["meta"]["conf"] = path
        conf["meta"]["name"] = file_name.split(".")[0]
        conf["meta"]["file_name"] = file_name

        if "parent" in conf["meta"]:
            parent = load_conf(conf["meta"]["parent"])
            conf = {**parent, **conf}

        if experiment_dir is not None:
            save_conf(experiment_dir+file_name, conf)

        f.close()

        return conf
