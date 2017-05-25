from os.path import join

import numpy as np
import cPickle as pickle
import os
import simplejson

from spodernet.utils.util import get_data_path, write_to_hdf, make_dirs_if_not_exists, load_hdf_file
from spodernet.interfaces import IAtBatchPreparedObservable
from spodernet.utils.global_config import Config

from spodernet.utils.logger import Logger
log = Logger('processors.py.txt')

class KeyToKeyMapper(IAtBatchPreparedObservable):
    def __init__(self, key2key):
        self.key2key = key2key

    def at_batch_prepared(self, batch_parts):
        str2var = batch_parts
        new_str2var = {}
        for key1, key2 in self.key2key.items():
            new_str2var[key2] = str2var[key1]

        return new_str2var

class TargetIdx2MultiTarget(IAtBatchPreparedObservable):
    def __init__(self, num_labels):
        self.num_labels = num_labels


    def at_batch_prepared(self, batch_parts):
        inp, inp_len, sup, sup_len, t, idx = batch_parts
        new_t = np.zeros((Config.batch_size, self.num_labels), dtype=np.int64)
        for i, row in enumerate(t):
            if len(t.shape) == 1:
                new_t[i, row] = 1
            else:
                for col in row:
                    new_t[i, col] = 1

        return [inp, inp_len, sup, sup_len, t, idx, new_t]


class ListIndexRemapper(object):
    def __init__(self, list_of_new_idx):
        self.list_of_new_idx = list_of_new_idx

    def at_batch_prepared(self, line):
        new_line = []
        for idx in self.list_of_new_idx:
            new_line.append(line[idx])

        return new_line

class JsonLoaderProcessors(object):
    def process(self, line):
        return simplejson.loads(line)

class RemoveLineOnJsonValueCondition(object):
    def __init__(self, key, func_condition):
        self.key = key
        self.func_condition = func_condition

    def process(self, json_dict):
        if self.func_condition(json_dict[self.key]):
            return None
        else:
            return json_dict

class DictKey2ListMapper(object):
    def __init__(self, ordered_keys_source):
        self.ordered_keys_source = ordered_keys_source

    def process(self, dict_object):
        list_of_ordered_values = []
        for key in self.ordered_keys_source:
            list_of_ordered_values.append(dict_object[key])
        return list_of_ordered_values


class AbstractProcessor(object):
    def __init__(self):
        self.state = None
        pass

    def link_with_pipeline(self, state):
        self.state = state

    def process(self, inputs, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractProcessor need to implement the process method')


class AbstractLoopLevelTokenProcessor(AbstractProcessor):
    def __init__(self):
        super(AbstractLoopLevelTokenProcessor, self).__init__()
        self.successive_for_loops_to_tokens = None

    def process_token(self, token, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractLoopLevelTokenProcessor need to implement the process_token method ')

    def process(self, sample, inp_type):
        if self.successive_for_loops_to_tokens == None:
            i = 0
            level = sample
            while not (   isinstance(level, basestring)
                       or isinstance(level, long)):
                    level = level[0]
                    i+=1
            self.successive_for_loops_to_tokens = i

        if self.successive_for_loops_to_tokens == 0:
            ret = self.process_token(sample, inp_type)

        elif self.successive_for_loops_to_tokens == 1:
            new_tokens = []
            for token in sample:
                new_tokens.append(self.process_token(token, inp_type))
            ret = new_tokens

        elif self.successive_for_loops_to_tokens == 2:
            new_sents = []
            for sent in sample:
                new_tokens = []
                for token in sent:
                    new_tokens.append(self.process_token(token, inp_type))
                new_sents.append(new_tokens)
            ret = new_sents

        return ret

class AbstractLoopLevelListOfTokensProcessor(AbstractProcessor):
    def __init__(self):
        super(AbstractLoopLevelListOfTokensProcessor, self).__init__()
        self.successive_for_loops_to_list_of_tokens = None

    def process_list_of_tokens(self, tokens, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractLoopLevelListOfTokensProcessor need to implement the process_list_of_tokens method ')

    def process(self, sample, inp_type):
        if self.successive_for_loops_to_list_of_tokens == None:
            i = 0
            level = sample
            while not (isinstance(level, basestring)
                       or isinstance(level, int)
                       or isinstance(level, np.int32)):
                    level = level[0]
                    i+=1
            self.successive_for_loops_to_list_of_tokens = i-1

        if self.successive_for_loops_to_list_of_tokens == 0:
            ret = self.process_list_of_tokens(sample, inp_type, samples_idx)

        elif self.successive_for_loops_to_list_of_tokens == 1:
            new_sents = []
            for sent in sample:
                new_sents.append(self.process_list_of_tokens(sent, inp_type))
            ret = new_sents

        return ret



class DeepSeqMap(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self, func):
        super(DeepSeqMap, self).__init__()
        self.func = func

    def process_list_of_tokens(self, data, inp_type):
        return self.func(data)

class Tokenizer(AbstractProcessor):
    def __init__(self, tokenizer_method):
        super(Tokenizer, self).__init__()
        self.tokenize = tokenizer_method

    def process(self, sentence, inp_type):
        return self.tokenize(sentence)

class NaiveNCharTokenizer(AbstractProcessor):
    def __init__(self, N=3):
        super(NaiveNCharTokenizer, self).__init__()
        self.N = N

    def process(self, sentence, inp_type):
        return [sentence[i:i+self.N] for i in range(0, len(sentence), self.N)]

class AddToVocab(AbstractLoopLevelTokenProcessor):
    def __init__(self, general_vocab_keys=['input', 'support']):
        super(AddToVocab, self).__init__()
        self.general_vocab_keys = set(general_vocab_keys)

    def process_token(self, token, inp_type):
        if inp_type == 'target':
            self.state['vocab']['general'].add_label(token)
            log.statistical('Example vocab target token {0}', 0.01, token)
        if inp_type in self.general_vocab_keys:
            self.state['vocab']['general'].add_token(token)
            message = 'Example vocab {0} token'.format(inp_type)
            log.statistical(message + ': {0}', 0.01, token)
        self.state['vocab'][inp_type].add_token(token)
        return token

class ToLower(AbstractProcessor):
    def __init__(self):
        super(ToLower, self).__init__()

    def process(self, token, inp_type):
        return token.lower()


class ConvertTokenToIdx(AbstractLoopLevelTokenProcessor):
    def __init__(self, keys2keys=None):
        super(ConvertTokenToIdx, self).__init__()
        self.keys2keys = keys2keys #maps key to other key, for example encode inputs with support vocabulary

    def process_token(self, token, inp_type):
        if not self.keys2keys is None and inp_type in self.keys2keys:
            return self.state['vocab'][self.keys2keys[inp_type]].get_idx(token)
        else:
            if inp_type != 'target':
                log.statistical('a non-label token {0}', 0.00001, token)
                return self.state['vocab']['general'].get_idx(token)
            else:
                log.statistical('a token {0}', 0.00001, token)
                return self.state['vocab']['general'].get_idx_label(token)

class ApplyFunction(AbstractProcessor):
    def __init__(self, func, keys=['input', 'support', 'target']):
        self.func = func
        self.keys = set(keys)

    def process(self, data, inp_type):
        if inp_type in self.keys:
            return self.func(data)
        else:
            return data

class SaveStateToList(AbstractProcessor):
    def __init__(self, name):
        super(SaveStateToList, self).__init__()
        self.name = name

    def link_with_pipeline(self, state):
        self.state = state
        if self.name not in self.state['data']:
            self.state['data'][self.name] = {}
        self.data = self.state['data'][self.name]

    def process(self, data, inp_type):
        if inp_type not in self.data: self.data[inp_type] = []
        self.data[inp_type].append(data)
        return data

class SaveLengthsToState(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self):
        super(SaveLengthsToState, self).__init__()

    def link_with_pipeline(self, state):
        self.state = state
        self.state['data']['lengths'] = {}
        self.data = self.state['data']['lengths']

    def process_list_of_tokens(self, tokens, inp_type):
        if inp_type not in self.data: self.data[inp_type] = []
        self.data[inp_type].append(int(len(tokens)))
        log.statistical('A list of tokens: {0}', 0.0001, tokens)
        log.debug_once('Pipeline {1}: A list of tokens: {0}', tokens, self.state['name'])
        return tokens

class StreamToHDF5(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self, name, samples_per_file=50000, keys=['input', 'support', 'target'], keys_for_length=['input', 'support']):
        super(StreamToHDF5, self).__init__()
        self.max_length = None
        self.samples_per_file = samples_per_file
        self.name = name
        self.idx = 0
        self.keys = keys
        if 'index' not in self.keys:
            self.keys.append('index')
        self.shard_id = {}
        self.max_lengths = {}
        self.data = {}
        for key in keys:
            self.shard_id[key] = 0
            self.max_lengths[key] = 0
            self.data[key] = []
        self.num_samples = None
        self.config = {'paths' : [], 'sample_count' : []}
        self.checked_for_lengths = False
        self.paths = {}
        self.shuffle_idx = None

    def link_with_pipeline(self, state):
        self.state = state
        self.base_path = join(self.state['path'], self.name)
        make_dirs_if_not_exists(self.base_path)

    def init_and_checks(self):
        if 'lengths' not in self.state['data']:
            log.error('Do a first pass to produce lengths first, that is use the "SaveLengths" ' \
                       'processor, execute, clean processors, then rerun the pipeline with hdf5 streaming.')
        if self.num_samples == None:
            self.num_samples = len(self.state['data']['lengths'][self.keys[0]])
        log.debug('Using type int32 for inputs and supports for now, but this may not be correct in the future')
        self.checked_for_lengths = True
        self.num_samples = len(self.state['data']['lengths'][self.keys[0]])
        log.debug('Number of samples as calcualted with the length data (SaveLengthsToState): {0}', self.num_samples)

    def process_list_of_tokens(self, tokens, inp_type):
        if not self.checked_for_lengths:
            self.init_and_checks()

        if self.max_lengths[inp_type] == 0:
            max_length = np.max(self.state['data']['lengths'][inp_type])
            log.debug('Calculated max length for input type {0} to be {1}', inp_type, max_length)
            self.max_lengths[inp_type] = max_length
            log.statistical('max length of the dataset: {0}', 0.0001, max_length)
        x = np.zeros((self.max_lengths[inp_type]), dtype=np.int32)
        x[:len(tokens)] = tokens
        if len(tokens) == 1 and self.max_lengths[inp_type] == 1:
            self.data[inp_type].append(x[0])
            log.debug_once('Adding one dimensional data for type ' + inp_type + ': {0}', x[0])
        else:
            self.data[inp_type].append(x.tolist())
            log.debug_once('Adding list data for type ' + inp_type + ': {0}', x.tolist())
        if inp_type == self.keys[-2]:
            self.data['index'].append(self.idx)
            self.idx += 1

        if (  len(self.data[inp_type]) == self.samples_per_file
           or len(self.data[inp_type]) == self.num_samples):
            self.save_to_hdf5(inp_type)


        if self.idx % 10000 == 0:
            if self.idx % 50000 == 0:
                log.info('Processed {0} samples so far...', self.idx)
            else:
                log.debug('Processed {0} samples so far...', self.idx)

        if self.idx == self.num_samples:
            counts = np.array(self.config['sample_count'])
            log.debug('Counts for each shard: {0}'.format(counts))
            fractions = counts / np.float32(np.sum(counts))
            self.config['fractions'] = fractions.tolist()
            self.config['counts'] = counts.tolist()
            self.config['paths'] = []
            for i in range(fractions.size):
                self.config['paths'].append(self.paths[i])

            pickle.dump(self.config, open(join(self.base_path, 'hdf5_config.pkl'), 'w'))

        return tokens

    def save_to_hdf5(self, inp_type):
        idx = self.shard_id[inp_type]
        X = np.array(self.data[inp_type], dtype=np.int32)
        file_name = inp_type + '_' + str(idx+1) + '.hdf5'
        if isinstance(X[0], list):
            new_X = []
            l = len(X[0])
            for i, list_item in enumerate(X):
                assert l == len(list_item)
            X = np.array(new_X, dtype=np.int32)
            log.debug("{0}", X)

        if inp_type == 'input':
            self.shuffle_idx = np.arange(X.shape[0])
            log.debug_once('First row of input data with shape {1} written to hdf5: {0}', X[0], X.shape)
            #X = X[self.shuffle_idx]
        log.debug('Writing hdf5 file for input type {0} to disk. Using index {1} and path {2}', inp_type, idx, join(self.base_path, file_name))
        log.debug('Writing hdf5 data. One sample row: {0}, shape: {1}, type: {2}', X[0], X.shape, X.dtype)
        write_to_hdf(join(self.base_path, file_name), X)
        if idx not in self.paths: self.paths[idx] = []
        self.paths[idx].append(join(self.base_path, file_name))


        if inp_type == self.keys[0]:
            log.statistical('Count of shard {0}; should be {1} most of the time'.format(X.shape[0], self.samples_per_file), 0.1)
            self.config['sample_count'].append(X.shape[0])

        if inp_type != self.keys[-2]:
            start = idx*self.samples_per_file
            end = (idx+1)*self.samples_per_file
            X_len = np.array(self.state['data']['lengths'][inp_type][start:end], dtype=np.int32)
            file_name_len = inp_type + '_lengths_' + str(idx+1) + '.hdf5'
            #X_len = X_len[self.shuffle_idx]
            write_to_hdf(join(self.base_path, file_name_len), X_len)
            self.paths[idx].append(join(self.base_path, file_name_len))
        else:
            file_name_index = 'index_' + str(idx+1) + '.hdf5'
            index = np.arange(self.idx - X.shape[0], self.idx, dtype=np.int32)
            #index = index[self.shuffle_idx]
            write_to_hdf(join(self.base_path, file_name_index), index)
            self.paths[idx].append(join(self.base_path, file_name_index))



        self.shard_id[inp_type] += 1
        del self.data[inp_type][:]



class CreateBinsByNestedLength(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self, name, min_batch_size=128, bins_of_same_length=True, raise_on_throw_away_fraction=0.2):
        super(CreateBinsByNestedLength, self).__init__()
        self.min_batch_size = min_batch_size
        self.pure_bins = bins_of_same_length
        self.raise_fraction = raise_on_throw_away_fraction
        self.length_key2bin_idx = {}
        self.performed_search = False
        self.name = name
        self.inp_type2idx = {'support' : 0, 'input' : 0, 'target' : 0}
        self.idx2data = {'support' : {}, 'input' : {}, 'target' : {}}
        self.binidx2data = {'support' : {}, 'input' : {}, 'index' : {}, 'target' : {}}
        self.binidx2bincount = {}
        self.binidx2numprocessed = {'support' : {}, 'input' : {}, 'target' : {}}

    def link_with_pipeline(self, state):
        self.state = state
        self.base_path = join(self.state['path'], self.name)
        self.temp_file_path = join(self.base_path, 'remaining_data.tmp')
        make_dirs_if_not_exists(self.base_path)

    def process_list_of_tokens(self, tokens, inp_type):
        if 'lengths' not in self.state['data']:
            log.error(('Do a first pass to produce lengths first, that is use the "SaveLengths" ',
                       'processor, execute, clean processors, then rerun the pipeline with this module.'))
        if inp_type not in self.state['data']['lengths']:
            log.error(('Do a first pass to produce lengths first, that is use the "SaveLengths" ',
                       'processor, execute, clean processors, then rerun the pipeline with this module.'))
        if not self.performed_search:
            self.perform_bin_search()

            assert (   isinstance(tokens[0], int)
                    or isinstance(tokens[0], np.int32)), \
                    'Token need to be either int or numpy int for binning to work!'

        idx = self.inp_type2idx[inp_type]
        self.idx2data[inp_type][idx] = tokens

        if inp_type == 'input' and idx % 10000 == 0:
            if idx % 50000 == 0:
                log.info('Processed {0} samples so far...', idx)
            else:
                log.debug('Processed {0} samples so far...', idx)

        if idx in self.idx2data['input'] and idx in self.idx2data['support'] and idx in self.idx2data['target']:
            x1 = self.idx2data['input'][idx]
            x2 = self.idx2data['support'][idx]
            t = self.idx2data['target'][idx]
            l1 = len(x1)
            l2 = len(x2)
            key = str(l1) + ',' + str(l2)
            if key not in self.length_key2bin_idx:
                self.inp_type2idx[inp_type] += 1
                return
            bin_idx = self.length_key2bin_idx[key]
            self.binidx2data['input'][bin_idx].append(np.array(x1, dtype=np.int32))
            self.binidx2data['support'][bin_idx].append(np.array(x2, dtype=np.int32))
            self.binidx2data['index'][bin_idx].append(idx)
            self.binidx2data['target'][bin_idx].append(np.array(t, dtype=np.int32))
            self.binidx2numprocessed[bin_idx] += 1
            self.inp_type2idx[inp_type] += 1

            if (len(self.binidx2data['input']) % 1000 == 0
               or (self.binidx2numprocessed[bin_idx] == self.binidx2bincount[bin_idx]
                         and len(self.binidx2data['input'][bin_idx]) > 0)):
                X_new = np.array(self.binidx2data['input'][bin_idx], dtype=np.int32)
                S_new = np.array(self.binidx2data['support'][bin_idx], dtype=np.int32)
                idx_new = np.array(self.binidx2data['index'][bin_idx], dtype=np.int32)
                t_new = np.array(self.binidx2data['target'][bin_idx], dtype=np.int32)
                if t_new.shape[1] == 1:
                    t_new = t_new.reshape(-1)

                pathX = join(self.base_path, 'input_bin_{0}.hdf5'.format(bin_idx))
                pathS = join(self.base_path, 'support_bin_{0}.hdf5'.format(bin_idx))
                pathX_len = join(self.base_path, 'input_lengths_bin_{0}.hdf5'.format(bin_idx))
                pathS_len = join(self.base_path, 'support_lengths_bin_{0}.hdf5'.format(bin_idx))
                pathIdx = join(self.base_path, 'index_bin_{0}.hdf5'.format(bin_idx))
                pathT = join(self.base_path, 'target_bin_{0}.hdf5'.format(bin_idx))

                if os.path.exists(pathX):
                    X_old = load_hdf_file(pathX)
                    S_old = load_hdf_file(pathS)
                    idx_old = load_hdf_file(pathIdx)
                    t_old = load_hdf_file(pathT)
                    X = np.vstack([X_old, X_new])
                    S = np.vstack([S_old, S_new])
                    index = np.vstack([idx_old, idx_new])
                    T = np.vstack([t_old, t_new])
                else:
                    X = X_new
                    S = S_new
                    index = idx_new
                    T = t_new

                write_to_hdf(pathX, X)
                write_to_hdf(pathS, S)
                write_to_hdf(pathX_len, np.ones((X.shape[0]), dtype=np.int32)*X.shape[1])
                write_to_hdf(pathS_len, np.ones((S.shape[0]), dtype=np.int32)*S.shape[1])
                write_to_hdf(pathIdx, index)
                write_to_hdf(pathT, T)
                del self.binidx2data['input'][bin_idx][:]
                del self.binidx2data['support'][bin_idx][:]
                del self.binidx2data['index'][bin_idx][:]
                del self.binidx2data['target'][bin_idx][:]

        else:
            self.inp_type2idx[inp_type] += 1

        return tokens


    def perform_bin_search(self):
        l1 = np.array(self.state['data']['lengths']['input'], dtype=np.int32)
        l2 = np.array(self.state['data']['lengths']['support'], dtype=np.int32)
        if self.pure_bins == False:
            raise NotImplementedError('Bin search currently only works for bins that feature samples of the same length')
        if self.pure_bins:
            self.wasted_lengths, self.length_tuple2bin_size = self.calculate_wastes(l1, l2)

        config = {'paths' : [], 'path2len' : {}, 'path2count' : {}}
        counts = []
        for i, ((l1, l2), count) in enumerate(self.length_tuple2bin_size):
            key = str(l1) + ',' + str(l2)
            self.length_key2bin_idx[key] = i
            self.binidx2data['input'][i] = []
            self.binidx2data['support'][i] = []
            self.binidx2data['index'][i] = []
            self.binidx2data['target'][i] = []
            self.binidx2numprocessed[i] = 0
            self.binidx2bincount[i] = count
            counts.append(count)
            pathX = join(self.base_path, 'input_bin_{0}.hdf5'.format(i))
            pathS = join(self.base_path, 'support_bin_{0}.hdf5'.format(i))
            pathX_len = join(self.base_path, 'input_lengths_bin_{0}.hdf5'.format(i))
            pathS_len = join(self.base_path, 'support_lengths_bin_{0}.hdf5'.format(i))
            pathIdx = join(self.base_path, 'index_bin_{0}.hdf5'.format(i))
            pathT = join(self.base_path, 'target_bin_{0}.hdf5'.format(i))
            config['paths'].append([pathX, pathX_len, pathS, pathS_len, pathT, pathIdx])
            config['path2len'][pathX] = l1
            config['path2len'][pathS] = l2
            config['path2count'][pathX] = count
            config['path2count'][pathS] = count

        config['fractions'] = (np.float64(np.array(counts)) / np.sum(counts))
        config['counts'] = counts
        self.config = config
        pickle.dump(config, open(join(self.base_path, 'hdf5_config.pkl'), 'w'))

        self.performed_search = True


    def calculate_wastes(self, l1, l2):
        wasted_samples = 0.0
        # get non-zero bin count, and the lengths corresponding to the bins
        counts_unfiltered = np.bincount(l1)
        lengths = np.arange(counts_unfiltered.size)
        counts = counts_unfiltered[counts_unfiltered > 0]
        lengths = lengths[counts_unfiltered > 0]
        indices = np.argsort(counts)
        # from smallest bin_counts to largest
        # look how many bins of l2 (support) are smaller than the min_batch_size
        wasted_lengths = []
        bin_by_size = []
        total_bin_count = 0.0
        for idx in indices:
            l1_waste = lengths[idx]
            l2_index = np.where(l1==l1_waste)[0]
            l2_counts_unfiltered = np.bincount(l2[l2_index])
            lengths2 = np.arange(l2_counts_unfiltered.size)
            l2_counts = l2_counts_unfiltered[l2_counts_unfiltered > 0]
            lengths2 = lengths2[l2_counts_unfiltered > 0]
            # keep track of the size of nested bins which will be included
            for length, bin_count in zip(lengths2, l2_counts):
                if bin_count >= self.min_batch_size:
                    bin_by_size.append(((l1_waste, length), bin_count))
                    total_bin_count += bin_count
            l2_waste = lengths2[l2_counts < self.min_batch_size]
            wasted_lengths.append([l1_waste, l2_waste])
            wasted_samples += np.sum(l2_counts[l2_counts < self.min_batch_size])
        wasted_fraction = wasted_samples / l1.size
        log.info('Wasted fraction for batch size {0} is {1}', self.min_batch_size, wasted_fraction)
        if wasted_fraction > self.raise_fraction:
            str_message = 'Wasted fraction {1}! This is higher than the raise error threshold of {0}!'.format(self.raise_fraction, wasted_fraction)
            log.error(str_message)
            raise Exception(str_message)

        # assign this here for testing purposes
        self.wasted_fraction = wasted_fraction
        self.total_bin_count = total_bin_count
        #self.num_samples = total_bin_count/(1.0-wasted_fraction)

        return wasted_lengths, bin_by_size





