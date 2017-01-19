# -*- coding: utf-8 -*-

# vim:set ft=python ts=4 sw=4 sts=4 autoindent:

'''
XXX: TODO:
Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2016-01-15
'''

# XXX: Not happy with the API yet... do we ever need a mutable variant?

from collections import OrderedDict


# XXX: This version is mutable!
# TODO: We need an immutable version with a set default!
class Identifier:
    def __init__(self, keys=None):
        self._key2id = OrderedDict()
        self._id2key = []
        if keys is not None:
            for k in keys:
                self[k]

    def __len__(self):
        return len(self._id2key)

    def __getitem__(self, key):
        try:
            return self._key2id[key]
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        _id = len(self._id2key)
        self._key2id[key] = _id
        self._id2key.append(key)
        return _id

    def key_by_id(self, _id):
        return self._id2key[_id]

    def __setitem__(self, key, _id):
        raise NotImplementedError('Assignment not supported')

    def __delitem__(self, key):
        raise NotImplementedError('Deletion not supported')

    def __iter__(self):
        return iter(self._key2id)

    def items(self):
        return self._key2id.items()

    def __contains__(self, key):
        return item in self._key2id

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__,
                ', '.join("{}: {}".format(repr(k), i) for k, i in self.items()))

    def __repr__(self):
        return '{}(({}))'.format(self.__class__.__name__,
            ', '.join(repr(k) for k in self))


class FrozenIdentifier(Identifier):
    def __init__(self, keys, default_key=None):
        super().__init__()
        self._default_key = default_key
        # TODO: Check if default_key is in the dict.
        if keys is not None:
            for i, k in enumerate(keys):
                self._key2id[k] = i
                self._id2key.append(k)

    def __getitem__(self, key):
        try:
            return self._key2id[key]
        except KeyError as e:
            if self._default_key is not None:
                return self._key2id[self._default_key]
            else:
                raise e
