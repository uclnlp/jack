# -*- coding: utf-8 -*-

import numpy as np
from quebap.sisyphos import batch


def test_get_buckets():
    #rs = np.random.RandomState(0)
    #data = rs.rand(2, 1024)
    data = {
        'data0': [i * [i] for i in range(1, 10)],
        'data1': [i * [i] for i in range(3, 12)]
    }

    buckets2ids, ids2buckets = batch.get_buckets(data=data,
                                                 order=('data0', 'data1'),
                                                 structure=(2, 2),
                                                 seed=0)

    assert buckets2ids == {'(1, 0)': [5, 6], '(1, 1)': [7, 8], '(0, 0)': [0, 1, 2], '(0, 1)': [3, 4]}
    assert ids2buckets == {0: '(0, 0)', 1: '(0, 0)', 2: '(0, 0)', 3: '(0, 1)', 4: '(0, 1)', 5: '(1, 0)', 6: '(1, 0)',
                           7: '(1, 1)', 8: '(1, 1)'}
