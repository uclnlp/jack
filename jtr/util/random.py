# -*- coding: utf-8 -*-

import numpy as np


def singleton(cls):
    instances = {}

    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return getinstance


@singleton
class DefaultRandomState(np.random.RandomState):
    def __init__(self, seed=None):
        super().__init__(seed)
