import numpy as np
import random
from itertools import islice

#from quebap.sisyphos.map import numpify


#make deterministic
SEED = 1337
np.random.seed(SEED)
random.seed(SEED)


def get_buckets(data, order, structure, seed=SEED):
    """
    :param data: sequence or dict of sequences in which entries of the inner sequence have the __len__ attribute
    :param order: tuple with highest-level indices (in case data is list) or keys (if data is dict) in data,
                  which need bucketing.
TODO: update documentation with dicts...
            e.g. data = [array1, array2, array3, array4]
                 and we want bucketing according to lengths of examples in array1 and array3
                 order = (0, 2) :  performs bucketing on array1, and within each bucket,
                                   again creates buckets according to array3
                                   (automatic bucketing will result in different array3 bucket boundaries
                                   within each bucket according to array1).
                                   Other indices (1 and 3) are ignored for bucketing.
                         (2, 0) :  vice versa, starting with array3 for the highest-level buckets
    :param structure: tuple with same length as `order`, each element is an integer or a list of integers
           For each position:
            - integer: denotes number of buckets, to be determined automatically
            - list: determines bucket boundaries. E.g.: [10, 20, 30] will result in 4 buckets
              (1) lengths 0-10, (2) lengths 11-20, (3) lengths 21-30, (4) lengths > 30
              e.g.: `order` = (0, 2) and `structure` = (3, [10]) generates 6 buckets:
              within each of 3 partition based on array1 lengths,
              there is a bucket with array2 instances of length 10 or less, and one for length > 10.
    :param seed: random seed
    :return: dict that maps instance-id (index along 1st dimension of elements in data) to bucket-id
             bucket-id's are tuples with same length as `order`.
    """
    #todo: asserts to check input arguments

    np.random.seed(int(10000*random.random())) #remains deterministic, but seems to be more random

    is_dict = isinstance(data,dict)
    n_tot = len(list(data.values())[0]) if is_dict else len(data[0])
    if order is None or structure is None:
        #all in 1 bucket, with id '(0)'
        buckets2ids = {'(0)':list(range(n_tot))}
        ids2buckets = dict(zip(list(range(n_tot)),['(0)']*n_tot))
        return buckets2ids, ids2buckets


    def _chunk(it, size):
        """returns iterator of chunks (tuples) from it (input iterator), with given size (last one may be shorter)"""
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    #todo: make faster if necessary
    def _partition(buckets2ids, _order, _structure):
        buckets2ids_new = {}
        for bid, ids in sorted(buckets2ids.items(), key=lambda x: x[0]):
            lengths = [len(data[_order[0]][id]) for id in ids]
            sorted_ids_lengths = sorted(zip(ids, lengths), key=lambda x: x[1])
            if isinstance(_structure[0], int):#automatic bucketing
                # how many buckets do we need to divide everything into
                # pieces of length structure[0]?
                size = len(lengths)//_structure[0] if len(lengths)%_structure[0]==0 else 1+len(lengths)//_structure[0]
                buckets = list(_chunk([tup[0] for tup in sorted_ids_lengths],size))
            else:# structure_now is sequence of ints
                struct = list(sorted(_structure[0]))+[np.inf]
                bin_max, struct = struct[0], struct[1:]
                buckets = [[]]
                for id,l in sorted_ids_lengths:
                    if l>bin_max: #never happens when bin_max = np.inf
                        bin_max, struct = struct[0], struct[1:]
                        buckets.append([])
                    buckets[-1].append(id)
            buckets2ids_new.update({tuple(list(bid)+[i]):list(bucket) for i,bucket in enumerate(buckets)})
        if len(_order)>1:
            buckets2ids_new = _partition(buckets2ids_new, _order[1:], _structure[1:])

        buckets2ids_new = {bid:bucket for bid,bucket in buckets2ids_new.items() if len(bucket)>0}
        return buckets2ids_new


    buckets2ids = _partition({():list(range(n_tot))}, order, structure)
    buckets2ids = {str(bid):buckets2ids[bid] for bid in buckets2ids} #make bucket-ids strings (for random.choice)

    ids2buckets = {}
    for bid,bucket in buckets2ids.items():
        ids2buckets.update({id:bid for id in bucket})

    return buckets2ids, ids2buckets




# todo: set seed
def get_batches(data, batch_size=32, pad=0, bucket_order=None, bucket_structure=None, batch_size_fixed=False, seed=SEED):
    """
    :param data: either a list of numpy arrays (or list of lists), or a dict with numpy arrays or lists
    :param batch_size: the desired batch size
    :param seed: random seed for shuffling
    :param pad: padding symbol in case data is list of lists of different sizes
    :param bucket_order: argument `order` in get_buckets (list with indices or keys); `None` if no bucketing
    :param bucket_structure: argument `structure` in get_buckets; `None` if no bucketing
    :param batch_size_fixed: if set to True, final batch (per bucket) will be ignored if < batch_size
    :return: returns a generator that generates: list or dict of [batch_size x _] 2D numpy tensors
    """

    is_dict = isinstance(data, dict)
    data0 = list(data.values())[0] if is_dict else data[0]
    if not isinstance(data0, np.ndarray):
        data_np = numpify(data,pad)#still need original data for length-based bucketing
    else:
        data_np = data

    def get_bucket_probs(_buckets2instances):
        N = float(np.sum([len(ids) for ids in _buckets2instances.values()]))
        return {bid: len(ids)/N if N>0. else 0. for bid,ids in _buckets2instances.items()}
    def shuffle_buckets(_buckets2instances):
        for bid in sorted(_buckets2instances.keys()):#sorted: to keep deterministic
            random.shuffle(_buckets2instances[bid])

    buckets2instances, _ = get_buckets(data, bucket_order, bucket_structure, seed=int(10000*random.random()))
    probs = get_bucket_probs(buckets2instances)

    def bucket_generator():
        buckets2instances, _ = get_buckets(data, bucket_order, bucket_structure)
        shuffle_buckets(buckets2instances)
        all_seen = False
        while not all_seen:
            bids,probs = zip(*sorted(get_bucket_probs(buckets2instances).items(), key=lambda x:x[0]))
            #sorted keys: to keep deterministic
            if np.sum(probs) == 0.:
                all_seen = True
            else:
                bid = np.random.choice(bids, replace=False, p=probs) #sample bucket according to remaining size
                batch_indices = buckets2instances[bid][:batch_size]
                buckets2instances[bid] = buckets2instances[bid][batch_size:]
                # if required by batch_size_fixed: ignore last batch in bucket if too small
                if not(batch_size_fixed and len(batch_indices)<batch_size):
                    if is_dict:
                        yield {k:data_np[k][batch_indices] for k in data_np}
                    else:
                        yield [x[batch_indices] for x in data_np]

    return GeneratorWithRestart(bucket_generator)


def get_feed_dicts(data, placeholders, batch_size=32, pad=0, bucket_order=None, bucket_structure=None, batch_size_fixed=False):
    """Creates feed dicts for all batches with a given batch size.

    Args:
        data (list or dict): The input data for the feed dicts.
        placeholders (list or dict): The TensorFlow placeholders for the data.
        batch_size (int): The batch size for the data.
        pad (int): Padding symbol index to pad lists of different sizes.
    Returns:
        GeneratorWithRestart: Generator that yields a feed_dict for each
        iteration. A feed dict consists of { name : tensor } key-value pairs.
    """
    def generator():
        batches = get_batches(data, batch_size, pad, bucket_order, bucket_structure, batch_size_fixed)
        # fixme: this is potentially inefficient as it might be called every time we retrieve a batch

        if isinstance(data, list) and isinstance(placeholders, list):
            mapped = map(lambda xs: dict(zip(placeholders, xs)), batches)
        elif isinstance(data, dict) and isinstance(placeholders, dict):
            #assume: all placeholders must be filled; potentially data has more fields
            assert set(placeholders.keys()).issubset(set(data.keys())), \
                'data keys %s \nnot compatible with placeholder keys %s'%(set(placeholders.keys()),set(data.keys()))
            mapped = map(lambda xs: {placeholders[k]: xs[k] for k in placeholders}, batches)
            #for each key in placeholders dict, pair the placeholder with the corresponding batch dict value
        else:
            raise IOError('incompatible types of data and model placeholders (need to be both lists, or both dicts)')

        for x in mapped:
            yield x

    return GeneratorWithRestart(generator)




class GeneratorWithRestart(object):
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return self.iterator()



#test bucketing
if __name__ == '__main__':
    import pprint
    from sisyphos.map import deep_seq_map
    pp = pprint.PrettyPrinter(indent=4)


    print('test bucketing')
    # print('(more general, but more difficult to track manually, when shuffling the data)')
    data0 = [i*[i] for i in range(1,10)]
    # #random.shuffle(data0)
    data1 = [i*[i] for i in range(3,12)]
    # #random.shuffle(data1)
    #data = deep_seq_map([data0,data1], lambda xs: len(xs), [0, 1], expand=True)
    data = deep_seq_map({'data0':data0,'data1':data1}, lambda xs: len(xs), keys=['data0', 'data1'], fun_name='len', expand=True)
    print('data keys:')
    pp.pprint(list(data.keys()))
    #order = (0,2)
    order = ('data0', 'data1')
    structure = (2,2)
    print('(1) create buckets with order=%s, structure=%s'%(str(order),str(structure)))
    buckets2ids, ids2buckets = get_buckets(data,order,structure)
    pp.pprint(buckets2ids)
    for id,bid in ids2buckets.items():
        if isinstance(data, dict):
            print('id: %d, bucket-id: %s, data: %s'%(id,str(bid),str({k: data[k][id] for k in data})))
        else:
            print('id: %d, bucket-id: %s, data: %s'%(id,str(bid),str([d[id] for d in data])))
    print('(2) test no bucketing, with order=None or structure=None; all in 1 bucket')
    buckets2ids, ids2buckets = get_buckets(data,None,None)
    pp.pprint(buckets2ids)
    for id,bid in ids2buckets.items():
        if isinstance(data, dict):
            print('id: %d, bucket-id: %s, data: %s'%(id,str(bid),str({k: data[k][id] for k in data})))
        else:
            print('id: %d, bucket-id: %s, data: %s'%(id,str(bid),str([d[id] for d in data])))


    order = ('data1','data0')
    structure = ([7],[7])
    print('(3) create buckets with order=%s, structure=%s' % (str(order), str(structure)))
    buckets2ids, ids2buckets = get_buckets(data, order, structure)
    pp.pprint(buckets2ids)
    for id, bid in sorted(ids2buckets.items(),key=lambda x:x[0]):
        if isinstance(data, dict):
            print('id: %d, bucket-id: %s, data: %s'%(id,str(bid),str({k: data[k][id] for k in data})))
        else:
            print('id: %d, bucket-id: %s, data: %s' % (id, str(bid), str([d[id] for d in data])))

    print('test batcher:')
    batcher = get_batches(data, batch_size=2, pad=0, bucket_order=order, bucket_structure=structure, batch_size_fixed=False)
    shown = 0
    while shown<2:
        print('new epoch:')
        for batch in batcher:
            pp.pprint(batch)
            pass
        shown+=1
