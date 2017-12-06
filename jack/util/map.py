# -*- coding: utf-8 -*-
import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_list_shape(xs):
    if isinstance(xs, int):
        shape = []
    else:
        shape = [len(xs)]
        for i, x in enumerate(xs):
            if isinstance(x, list) or isinstance(x, tuple):
                if len(shape) == 1:
                    shape.append(0)
                shape[1] = max(len(x), shape[1])
                for j, y in enumerate(x):
                    if isinstance(y, list):
                        if len(shape) == 2:
                            shape.append(0)
                        shape[2] = max(len(y), shape[2])
    return shape


def numpify(xs, pad=0, keys=None, dtypes=None):
    """Converts a dict or list of Python data into a dict of numpy arrays."""
    is_dict = isinstance(xs, dict)
    xs_np = {} if is_dict else [0] * len(xs)
    xs_iter = xs.items() if is_dict else enumerate(xs)

    for i, (key, x) in enumerate(xs_iter):
        try:
            if (keys is None or key in keys) and not isinstance(x, np.ndarray):
                shape = get_list_shape(x)
                dtype = dtypes[i] if dtypes is not None else np.int64
                x_np = np.full(shape, pad, dtype)

                nb_dims = len(shape)

                if nb_dims == 0:
                    x_np = x
                else:
                    def f(tensor, values):
                        t_shp = tensor.shape
                        if len(t_shp) > 1:
                            for _i, _values in enumerate(values):
                                f(tensor[_i], _values)
                        else:
                            tensor[0:len(values)] = [v for v in values]

                    f(x_np, x)

                xs_np[key] = x_np
            else:
                xs_np[key] = x
        except Exception as e:
            logger.error('Error numpifying value ' + str(x) + ' of key ' + str(key))
            raise e
    return xs_np
