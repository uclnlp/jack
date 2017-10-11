# -*- coding: utf-8 -*-

from jack.util import batch


def test_get_buckets():
    data = {
        'data0': [i * [i] for i in range(1, 10)],
        'data1': [i * [i] for i in range(3, 12)]
    }

    buckets2ids, ids2buckets = batch.get_buckets(data=data,
                                                 order=('data0', 'data1'),
                                                 structure=(2, 2))

    assert buckets2ids == {
        '(1, 0)': [5, 6],
        '(1, 1)': [7, 8],
        '(0, 0)': [0, 1, 2],
        '(0, 1)': [3, 4]
    }
    assert ids2buckets == {
        0: '(0, 0)',
        1: '(0, 0)',
        2: '(0, 0)',
        3: '(0, 1)',
        4: '(0, 1)',
        5: '(1, 0)',
        6: '(1, 0)',
        7: '(1, 1)',
        8: '(1, 1)'
    }


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
