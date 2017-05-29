# -*- coding: utf-8 -*-

import numpy as np

from jtr.util import batch


def test_get_buckets():
    data = {
        'data0': [i * [i] for i in range(1, 10)],
        'data1': [i * [i] for i in range(3, 12)]
    }

    buckets2ids, ids2buckets = batch.get_buckets(data=data,
                                                 order=('data0', 'data1'),
                                                 structure=(2, 2))

    assert buckets2ids == {'(1, 0)': [5, 6], '(1, 1)': [7, 8], '(0, 0)': [0, 1, 2], '(0, 1)': [3, 4]}
    assert ids2buckets == {0: '(0, 0)', 1: '(0, 0)', 2: '(0, 0)', 3: '(0, 1)', 4: '(0, 1)', 5: '(1, 0)', 6: '(1, 0)',
                           7: '(1, 1)', 8: '(1, 1)'}


def test_get_batches():
    data = {
        'data0': [[i] * 2 for i in range(10)],
        'data1': [[i] * 3 for i in range(10)]
    }

    batch_generator = batch.get_batches(data, batch_size=3, exact_epoch=True)
    batches = list(batch_generator)

    assert batches[0]['data0'].shape == batches[1]['data0'].shape == batches[2]['data0'].shape == (3, 2)
    assert batches[0]['data1'].shape == batches[1]['data1'].shape == batches[2]['data1'].shape == (3, 3)

    assert batches[3]['data0'].shape == (1, 2)
    assert batches[3]['data1'].shape == (1, 3)

    assert len(batches) == 4

    batch_generator = batch.get_batches(data, batch_size=3, exact_epoch=False)
    batches = list(batch_generator)

    assert len(batches) == 3


def test_get_feed_dicts():
    data = {
        'data0': [[i] * 2 for i in range(5)],
        'data1': [[i] * 3 for i in range(5)]
    }

    placeholders = {
        'data0': 'X',
        'data1': 'Y'
    }

    feed_dicts_it = batch.get_feed_dicts(data, placeholders, batch_size=2, exact_epoch=True)
    feed_dicts = list(feed_dicts_it)

    assert feed_dicts[0]['X'].shape == feed_dicts[1]['X'].shape == (2, 2)
    assert feed_dicts[2]['X'].shape == (1, 2)

    assert feed_dicts[0]['Y'].shape == feed_dicts[1]['Y'].shape == (2, 3)
    assert feed_dicts[2]['Y'].shape == (1, 3)

    for i in [0, 1, 2]:
        assert feed_dicts[i]['X'][0, 0] == feed_dicts[i]['Y'][0, 0]

    assert len(feed_dicts) == 3

    feed_dicts_it = batch.get_feed_dicts(data, placeholders, batch_size=2, exact_epoch=False)
    feed_dicts = list(feed_dicts_it)

    assert len(feed_dicts) == 2
