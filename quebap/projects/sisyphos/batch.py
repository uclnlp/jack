import numpy as np
from random import shuffle

from sisyphos.map import numpify


# todo: flag for filling up last batch
# todo: bucketing
def get_batches(data, batch_size=32, pad=0):
    """
    :param data: either a list of numpy arrays or a list of lists of examples
    :param batch_size: the desired batch size
    :param seed: random seed for shuffling
    :param PAD: padding symbol in case data is list of lists of different sizes
    :return: returns a generator of list of [batch_size x _] 2D numpy tensors
    """
    if not isinstance(data[0], np.ndarray):
        data = numpify(data, pad)

    def generator():
        indices = list(range(0, data[0].__len__()))
        shuffle(indices)
        num_iter = indices.__len__() // batch_size

        for i in range(num_iter):
            batch_indices = indices[:batch_size]
            indices = indices[batch_size:]
            yield [x[batch_indices] for x in data]

        if indices.__len__() > 0:
            yield [x[indices] for x in data]

    return GeneratorWithRestart(generator)


def get_feed_dicts(data, placeholders, batch_size=32, pad=0):
    def generator():
        batches = get_batches(data, batch_size, pad)
        # fixme: this is potentially inefficient as it might be called every
        # time we retrieve a batch
        mapped = map(lambda xs: dict(zip(placeholders, xs)), batches)
        for x in mapped:
            yield x

    return GeneratorWithRestart(generator)


class GeneratorWithRestart(object):
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return self.iterator()
