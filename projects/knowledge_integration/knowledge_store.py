import os
import pickle
import shelve

import spacy


class KnowledgeStore(object):
    def __init__(self, path, writeback=False):
        self._path = path
        self._sws = spacy.en.STOP_WORDS
        self._assertion_db = dict()
        self._object2assertions = dict()
        self._subject2assertions = dict()
        self._num_assertions = 0
        self._writeback = writeback

        self._assertion_cache = dict()

        if os.path.exists(os.path.join(path, 'object2assertions')):
            for fn in os.listdir(os.path.join(path, 'object2assertions')):
                with open(os.path.join(path, 'object2assertions', fn), 'rb') as f:
                    self._object2assertions[fn] = pickle.load(f)
            for fn in os.listdir(os.path.join(path, 'subject2assertions')):
                with open(os.path.join(path, 'subject2assertions', fn), 'rb') as f:
                    self._subject2assertions[fn] = pickle.load(f)
            for fn in os.listdir(os.path.join(path, 'assertions')):
                self._assertion_db[fn] = shelve.open(
                    os.path.join(path, 'assertions', fn), flag='c' if writeback else 'r', writeback=writeback)
                self._assertion_cache[fn] = dict()
                self._num_assertions += len(self._assertion_db[fn])
        else:
            os.makedirs(os.path.join(path, 'object2assertions'))
            os.makedirs(os.path.join(path, 'subject2assertions'))
            os.makedirs(os.path.join(path, 'assertions'))

    def get_connecting_assertion_keys(self, source_tokens, target_tokens, resources):
        """Returns: mapping from assertion keys to IDF scores."""

        def key_iterator(tokens):
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + 6, len(tokens) + 1)):
                    if tokens[j - 1] not in self._sws and tokens[j - 1].isalnum():
                        yield tokens[i:j], i, j

        source_obj_assertions = dict()
        source_subj_assertions = dict()
        keys = set()
        for ks, start, end in key_iterator(source_tokens):
            k = ' '.join(ks)
            if k in keys:
                continue
            keys.add(k)
            for source in resources:
                k_assertions = self._object2assertions[source].get(k)
                if k_assertions is not None:
                    idf = 1.0 / len(k_assertions)
                    for a in k_assertions:
                        source_obj_assertions[a] = (
                            max(source_obj_assertions.get(a, (0.0, None))[0], idf), ks, start, end)
                k_assertions = self._subject2assertions[source].get(k)
                if k_assertions is not None:
                    idf = 1.0 / len(k_assertions)
                    for a in k_assertions:
                        source_subj_assertions[a] = (
                            max(source_subj_assertions.get(a, (0.0, None))[0], idf), ks, start, end)

        assertions = dict()
        assertion_args = dict()
        keys = set()
        for ks, start, end in key_iterator(target_tokens):
            k = ' '.join(ks)
            if k in keys:
                continue
            keys.add(k)
            for source in resources:
                # subject from target, object from source
                k_assertions_subj = self._subject2assertions[source].get(k)
                if k_assertions_subj is not None:
                    idf2 = 1.0 / len(k_assertions_subj)
                    for a in k_assertions_subj:
                        idf, ks2, start2, end2 = source_obj_assertions.get(a, (None, None, None, None))
                        if idf is None or all(k in ks2 for k in ks) or all(k in ks for k in ks2):
                            continue
                        assertions[a] = max(assertions.get(a, 0.0), idf * idf2)
                        assertion_args[a] = [start2, end2], [start, end]
                # subject from source, object from target
                k_assertions = self._object2assertions[source].get(k)
                if k_assertions is not None:
                    idf2 = 1.0 / len(k_assertions)
                    for a in k_assertions:
                        idf, ks2, start2, end2 = source_subj_assertions.get(a, (None, None, None, None))
                        if idf is None or all(k in ks2 for k in ks) or all(k in ks for k in ks2):
                            continue
                        assertions[a] = max(assertions.get(a, 0.0), idf * idf2)
                        assertion_args[a] = [start2, end2], [start, end]
        return assertions, assertion_args

    def get_assertion_keys(self, tokens, resources):
        """Returns: mapping from assertion keys to IDF scores."""

        def key_iterator(tokens):
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + 6, len(tokens) + 1)):
                    if tokens[j - 1] not in self._sws and tokens[j - 1].isalnum():
                        yield tokens[i:j], i, j

        assertions = dict()
        assertion_args = dict()
        keys = set()
        for ks, start, end in key_iterator(tokens):
            k = ' '.join(ks)
            if k in keys:
                continue
            keys.add(k)
            for source in resources:
                k_assertions = self._object2assertions[source].get(k)
                if k_assertions is not None:
                    idf = 1.0 / len(k_assertions)
                    for a in k_assertions:
                        assertions[a] = max(assertions.get(a, 0.0), idf)
                        assertion_args[a] = [start, end]
                k_assertions = self._subject2assertions[source].get(k)
                if k_assertions is not None:
                    idf = 1.0 / len(k_assertions)
                    for a in k_assertions:
                        assertions[a] = max(assertions.get(a, 0.0), idf)
                        assertion_args[a] = [start, end]

        return assertions, assertion_args

    def assertion_keys_for_subject(self, subj, resource='default'):
        return self._subject2assertions[resource].get(subj, set())

    def assertion_keys_for_object(self, subj, resource='default'):
        return self._object2assertions[resource].get(subj, set())

    def get_assertion(self, assertion_key, cache=False):
        resource = assertion_key[:assertion_key.index('$')]
        ret = self._assertion_cache[resource].get(assertion_key)
        if ret is None:
            ret = self._assertion_db[resource].get(assertion_key)
            if cache:
                self._assertion_cache[resource][assertion_key] = ret
        return ret

    def add_assertion(self, assertion, subjects, objects, resource='default', key=None):
        assert '$' not in resource
        key = resource + '$' + (key or str(self._num_assertions))
        if resource not in self._object2assertions:
            self._object2assertions[resource] = dict()
            self._subject2assertions[resource] = dict()
            self._assertion_db[resource] = shelve.open(
                os.path.join(self._path, 'assertions', resource), flag='c' if self._writeback else 'r',
                writeback=self._writeback)
            self._assertion_cache[resource] = dict()
        o2a = self._object2assertions[resource]
        s2a = self._subject2assertions[resource]
        for o in objects:
            if o not in o2a:
                o2a[o] = set()
            o2a[o].add(key)
        for s in subjects:
            if s not in s2a:
                s2a[s] = set()
            s2a[s].add(key)
        self._assertion_db[resource][key] = assertion
        self._num_assertions += 1

    def save(self):
        for key in self._object2assertions:
            with open(os.path.join(self._path, 'object2assertions', key), 'wb') as f:
                pickle.dump(self._object2assertions[key], f)
            with open(os.path.join(self._path, 'subject2assertions', key), 'wb') as f:
                pickle.dump(self._subject2assertions[key], f)
            self._assertion_db[key].sync()
