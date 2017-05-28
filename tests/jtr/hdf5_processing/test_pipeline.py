import itertools
import json
import os
import pickle
import shutil
import uuid
from os.path import join

import nltk
import numpy as np
import pytest
import scipy.stats
from jtr.util.hdf5_processing.hooks import LossHook, AccuracyHook
from jtr.util.hdf5_processing.pipeline import Pipeline
from jtr.util.hdf5_processing.processors import JsonLoaderProcessors, RemoveLineOnJsonValueCondition, \
    DictKey2ListMapper
from jtr.util.hdf5_processing.processors import StreamToHDF5, CreateBinsByNestedLength, DeepSeqMap
from jtr.util.hdf5_processing.processors import Tokenizer, SaveStateToList, AddToVocab, ToLower, \
    ConvertTokenToIdx, SaveLengthsToState
from jtr.util.hdf5_processing.vocab import Vocab

from jtr.util.global_config import Config, Backends
from jtr.util.hdf5_processing.batching import StreamBatcher, BatcherState
from jtr.util.util import get_data_path, load_hdf_file

Config.backend = Backends.TEST

def get_test_data_path_dict():
    paths = {}
    paths['snli'] = './tests/test_data/snli.json'
    paths['snli3k'] = './tests/test_data/snli_3k.json'
    paths['snli1k'] = './tests/test_data/snli_1k.json'
    paths['wiki'] = './tests/test_data/wiki.json'

    return paths

def test_dict2listmapper():
    with open(join(get_data_path(), 'test.txt'), 'w') as f:
        for i in range(10):
            test_dict = {}
            test_dict['key1'] = str(i+5)
            test_dict['key2'] = str(i+3)
            test_dict['key3'] = str(i+4)
            f.write(json.dumps(test_dict) + '\n')

    p = Pipeline('abc')
    p.add_path(join(get_data_path(), 'test.txt'))
    p.add_line_processor(JsonLoaderProcessors())
    p.add_line_processor(DictKey2ListMapper(['key3', 'key1', 'key2']))
    p.add_text_processor(SaveStateToList('lines'))
    state = p.execute()
    for i, line in enumerate(state['data']['lines']['input']):
        assert int(line) == i+4, 'Input values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['support']):
        assert int(line) == i+5, 'Support values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['target']):
        assert int(line) == i+3, 'Target values does not correspond to the json key mapping.'

    os.remove(join(get_data_path(), 'test.txt'))
    shutil.rmtree(join(get_data_path(), 'abc'))

def test_remove_on_json_condition():
    with open(join(get_data_path(), 'test.txt'), 'w') as f:
        for i in range(10):
            test_dict = {}
            test_dict['key1'] = str(i+5)
            test_dict['key2'] = str(i+3)
            test_dict['key3'] = str(i+4)
            f.write(json.dumps(test_dict) + '\n')
            test_dict = {}
            test_dict['key1'] = str(i+5)
            test_dict['key2'] = str(i+3)
            test_dict['key3'] = 'remove me'
            f.write(json.dumps(test_dict) + '\n')

    p = Pipeline('abc')
    p.add_path(join(get_data_path(), 'test.txt'))
    p.add_line_processor(JsonLoaderProcessors())
    p.add_line_processor(RemoveLineOnJsonValueCondition('key3', lambda inp: inp == 'remove me'))
    p.add_line_processor(DictKey2ListMapper(['key3', 'key1', 'key2']))
    p.add_text_processor(SaveStateToList('lines'))
    state = p.execute()

    assert len(state['data']['lines']['input']) == 10, 'Length different from filtered length!'
    for i, line in enumerate(state['data']['lines']['input']):
        assert int(line) == i+4, 'Input values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['support']):
        assert int(line) == i+5, 'Support values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['target']):
        assert int(line) == i+3, 'Target values does not correspond to the json key mapping.'

    os.remove(join(get_data_path(), 'test.txt'))
    shutil.rmtree(join(get_data_path(), 'abc'))


def test_tokenization():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_sent_processor(SaveStateToList('tokens'))
    state = p.execute()

    inp_sents = state['data']['tokens']['input']
    sup_sents = state['data']['tokens']['support']
    sents = inp_sents + sup_sents

    # 2. setup nltk tokenization
    with open(get_test_data_path_dict()['snli']) as f:
        tokenized_sents = {'input' : [], 'support' : []}
        for line in f:
            inp, sup, t = json.loads(line)
            tokenized_sents['input'].append(tokenizer.tokenize(inp))
            tokenized_sents['support'].append(tokenizer.tokenize(sup))

    sents_nltk = tokenized_sents['input'] + tokenized_sents['support']
    # 3. test equality
    assert len(sents) == len(sents_nltk), 'Sentence count differs!'
    for sent1, sent2 in zip(sents, sents_nltk):
        assert len(sent1) == len(sent2), 'Token count differs!'
        for token1, token2 in zip(sent1, sent2):
            assert token1 == token2, 'Token values differ!'

def test_path_creation():
    names = []
    for i in range(100):
        names.append(str(uuid.uuid4()))

    for name in names:
        p = Pipeline(name)

    home = os.environ['HOME']
    paths = [ join(home, '.data', name) for name in names]
    for path in paths:
        assert os.path.exists(path)
        os.rmdir(path)

def test_vocab():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    state = p.execute()

    # 1. use Vocab manually and test it against manual vocabulary
    idx2token = {}
    token2idx = {}
    token2idx['OOV'] = 0
    idx2token[0] = 'OOV'
    # empty = 0
    token2idx[''] = 1
    idx2token[1] = ''
    idx = 2
    v = Vocab('test')
    with open(get_test_data_path_dict()['snli']) as f:
        tokenized_sents = {'input' : [], 'support' : []}
        for line in f:
            inp, sup, t = json.loads(line)

            for token in tokenizer.tokenize(inp):
                v.add_token(token)
                if token not in token2idx:
                    token2idx[token] = idx
                    idx2token[idx] = token
                    idx += 1

            for token in tokenizer.tokenize(sup):
                v.add_token(token)
                if token not in token2idx:
                    token2idx[token] = idx
                    idx2token[idx] = token
                    idx += 1

            v.add_label(t)


    # 3. Compare vocabs
    v2 = state['vocab']['general']
    for token in v.token2idx:
        assert v.token2idx[token] == v2.token2idx[token], 'Index for token not the same!'
        assert v.token2idx[token] == token2idx[token], 'Index for token not the same!'

    for idx in v.idx2token:
        assert v.idx2token[idx] == v2.idx2token[idx], 'Token for index not the same!'
        assert v.idx2token[idx] == idx2token[idx], 'Token for index not the same!'

    for label in v.label2idx:
        assert v.label2idx[label] == v2.label2idx[label], 'Index for label not the same!'

    for idx in v.idx2label:
        assert v.idx2label[idx] == v2.idx2label[idx], 'Label for index not the same!'


def test_separate_vocabs():

    # 1. write test data
    file_path = join(get_data_path(), 'test_pipeline', 'test_data.json')
    with open(file_path, 'w') as f:
        f.write(json.dumps(['0', 'a','-']) + '\n')
        f.write(json.dumps(['1', 'b','&']) + '\n')
        f.write(json.dumps(['2', 'c','#']) + '\n')

    # 2. read test data with pipeline
    p = Pipeline('test_pipeline')

    p.add_path(file_path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_token_processor(AddToVocab())
    state = p.execute()
    vocab = state['vocab']['general']
    inp_vocab = state['vocab']['input']
    sup_vocab = state['vocab']['support']
    tar_vocab = state['vocab']['target']

    # 6 token + empty and unknown = 8 
    assert vocab.num_token == 6 + 2, 'General vocab token count should be 8, but was {0} instead.'.format(vocab.num_token)
    assert vocab.num_labels == 3, 'General vocab token count should be 3, but was {0} instead.'.format(vocab.num_labels)

    assert inp_vocab.num_token == 3 + 2, 'General vocab token count should be 5, but was {0} instead.'.format(inp_vocab.num_token)
    assert inp_vocab.num_labels == 0, 'General vocab token count should be 0, but was {0} instead.'.format(inp_vocab.num_labels)
    assert sup_vocab.num_token == 3 + 2, 'General vocab token count should be 5, but was {0} instead.'.format(sup_vocab.num_token)
    assert sup_vocab.num_labels == 0, 'General vocab token count should be 0, but was {0} instead.'.format(sup_vocab.num_labels)
    assert tar_vocab.num_token == 3 + 2, 'General vocab token count should be 5, but was {0} instead.'.format(tar_vocab.num_token)
    assert tar_vocab.num_labels == 0, 'General vocab token count should be 0, but was {0} instead.'.format(tar_vocab.num_labels)

    for token in ['0', '1', '2']:
        assert token in vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)
        assert token in inp_vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)

    for token in ['a', 'b', 'c']:
        assert token in vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)
        assert token in sup_vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)

    for token in ['-', '&', '#']:
        assert token in vocab.label2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)
        assert token in tar_vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)


def test_to_lower_sent():
    path = get_test_data_path_dict()['snli']

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(ToLower())
    p.add_sent_processor(SaveStateToList('sents'))
    state = p.execute()

    inp_sents = state['data']['sents']['input']
    sup_sents = state['data']['sents']['support']
    sents = inp_sents + sup_sents

    # 2. test lowercase
    assert len(sents) == 200 # we have 100 samples for snli
    for sent in sents:
        assert sent == sent.lower(), 'Sentence is not lower case'

def test_to_lower_token():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    path = get_test_data_path_dict()['snli']

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(ToLower())
    p.add_token_processor(SaveStateToList('tokens'))
    state = p.execute()

    inp_tokens = state['data']['tokens']['input']
    sup_tokens = state['data']['tokens']['support']
    tokens = inp_tokens + sup_tokens

    # 2. test lowercase
    for token in tokens:
        assert token == token.lower(), 'Token is not lower case'

def test_save_to_list_text():
    path = get_test_data_path_dict()['wiki']

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_text_processor(SaveStateToList('text'))
    state = p.execute()

    inp_texts = state['data']['text']['input']
    sup_texts = state['data']['text']['support']
    assert len(inp_texts) == 3, 'The input data size should be three samples, but found {0}'.format(len(inp_texts))
    assert len(inp_texts) == 3, 'The input data size should be three samples, but found {0}'.format(len(sup_texts))
    with open(path) as f:
        for inp1, sup1, line in zip(inp_texts, sup_texts, f):
            inp2, sup2, t = json.loads(line)
            assert inp1 == inp2, 'Saved text data not the same!'
            assert sup1 == sup2, 'Saved text data not the same!'


def test_save_to_list_sentences():
    path = get_test_data_path_dict()['wiki']
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_text_processor(Tokenizer(sent_tokenizer.tokenize))
    p.add_sent_processor(SaveStateToList('sents'))
    state = p.execute()

    # 2. setup manual sentence processing
    inp_sents = state['data']['sents']['input']
    sup_sents = state['data']['sents']['support']
    inp_sents2 = []
    sup_sents2 = []
    with open(path) as f:
        for line in f:
            inp, sup, t = json.loads(line)
            sup_sents2 += sent_tokenizer.tokenize(sup)
            inp_sents2 += sent_tokenizer.tokenize(inp)

    # 3. test equivalence
    assert len(inp_sents) == len(inp_sents2), 'Sentence count differs!'
    assert len(sup_sents) == len(sup_sents2), 'Sentence count differs!'

    for sent1, sent2 in zip(inp_sents, inp_sents2):
        assert sent1 == sent2, 'Saved sentence data not the same!'

    for sent1, sent2 in zip(sup_sents, sup_sents2):
        assert sent1 == sent2, 'Saved sentence data not the same!'


def test_save_to_list_post_process():
    path = get_test_data_path_dict()['wiki']
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_text_processor(Tokenizer(sent_tokenizer.tokenize))
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveStateToList('samples'))
    state = p.execute()

    # 2. setup manual sentence -> token processing
    inp_samples = state['data']['samples']['input']
    sup_samples = state['data']['samples']['support']
    inp_samples2 = []
    sup_samples2 = []
    with open(path) as f:
        for line in f:
            sup_sents = []
            inp_sents = []
            inp, sup, t = json.loads(line)
            for sent in sent_tokenizer.tokenize(sup):
                sup_sents.append(tokenizer.tokenize(sent))
            for sent in sent_tokenizer.tokenize(inp):
                inp_sents.append(tokenizer.tokenize(sent))
            inp_samples2.append(inp_sents)
            sup_samples2.append(sup_sents)


    # 3. test equivalence
    for sample1, sample2 in zip(inp_samples, inp_samples2):
        assert len(sample1) == len(sample2), 'Sentence count differs!'
        for sent1, sent2, in zip(sample1, sample2):
            assert len(sent1) == len(sent2), 'Token count differs!'
            for token1, token2 in zip(sent1, sent2):
                assert token1 == token2, 'Tokens differ!'

    for sample1, sample2 in zip(sup_samples, sup_samples2):
        assert len(sample1) == len(sample2), 'Sentence count differs!'
        for sent1, sent2, in zip(sample1, sample2):
            assert len(sent1) == len(sent2), 'Token count differs!'
            for token1, token2 in zip(sent1, sent2):
                assert token1 == token2, 'Tokens differ!'



def test_convert_token_to_idx_no_sentences():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    state = p.execute()

    inp_indices = state['data']['idx']['input']
    label_idx = state['data']['idx']['target']

    # 2. use Vocab manually
    v = Vocab('test')
    with open(get_test_data_path_dict()['snli']) as f:
        for line in f:
            inp, sup, t = json.loads(line)

            for token in tokenizer.tokenize(inp):
                v.add_token(token)

            for token in tokenizer.tokenize(sup):
                v.add_token(token)

            v.add_label(t)

    # 3. index manually
    with open(get_test_data_path_dict()['snli']) as f:
        tokenized_sents = {'input' : [], 'support' : [], 'target' : []}
        for line in f:
            inp_idx = []
            sup_idx = []

            inp, sup, t = json.loads(line)

            for token in tokenizer.tokenize(inp):
                inp_idx.append(v.get_idx(token))

            for token in tokenizer.tokenize(sup):
                sup_idx.append(v.get_idx(token))

            tokenized_sents['target'].append(v.get_idx_label(t))
            tokenized_sents['input'].append(inp_idx)
            tokenized_sents['support'].append(sup_idx)


    # 4. Compare idx
    assert len(tokenized_sents['input']) == len(inp_indices), 'Sentence count differs!'
    for sent1, sample in zip(tokenized_sents['input'], inp_indices):
        sent2 = sample[0] # in this case we do not have sentences
        assert len(sent1) == len(sent2), 'Index count (token count) differs!'
        for idx1, idx2 in zip(sent1, sent2):
            assert idx1 == idx2, 'Index for token differs!'

    # 5. Compare label idx
    for idx1, sample in zip(tokenized_sents['target'], label_idx):
        # sample[0] == sent
        # sent[0] = idx
        assert idx1 == sample[0][0], 'Index for label differs!'


def test_convert_to_idx_with_separate_vocabs():

    # 1. write test data
    file_path = join(get_data_path(), 'test_pipeline', 'test_data.json')
    with open(file_path, 'w') as f:
        f.write(json.dumps(['0', 'a','-']) + '\n')
        f.write(json.dumps(['1', 'b','&']) + '\n')
        f.write(json.dumps(['2', 'c','#']) + '\n')

    # 2. read test data with pipeline
    keys2keys = {}
    keys2keys['input'] = 'input'
    keys2keys['support'] = 'support'

    p = Pipeline('test_pipeline')
    p.add_path(file_path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys))
    p.add_post_processor(SaveStateToList('idx'))
    state = p.execute()

    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['input']

    # 0 = UNK, 1 = '', 2,3,4 -> max index is 4
    assert np.max(inp_indices) == 2 + 2, 'Max idx should have been 2 if the vocabularies were separates!'
    assert np.max(sup_indices) == 2 + 2, 'Max idx should have been 2 if the vocabularies were separates!'

def test_save_lengths():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveLengthsToState())
    state = p.execute()

    lengths_inp = state['data']['lengths']['input']
    lengths_sup = state['data']['lengths']['support']
    lengths1 = lengths_inp + lengths_sup

    # 2. generate lengths manually
    lengths_inp2 = []
    lengths_sup2 = []
    with open(get_test_data_path_dict()['snli']) as f:
        for line in f:
            inp, sup, t = json.loads(line)

            lengths_inp2.append(len(tokenizer.tokenize(inp)))
            lengths_sup2.append(len(tokenizer.tokenize(sup)))

    lengths2 = lengths_inp2 + lengths_sup2

    # 3. test for equal lengths
    assert len(lengths1) == len(lengths2), 'Count of lengths differs!'
    assert len(lengths1) == 200, 'Count of lengths not as expected for SNLI test data!'
    for l1, l2 in zip(lengths1, lengths2):
        assert l1 == l2, 'Lengths of sentence differs!'


def test_stream_to_hdf5():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'snli_test'
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder, data_folder_name)
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline(pipeline_folder)
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveLengthsToState())
    p.execute()
    p.clear_processors()

    # 2. Process the data further to stream it to hdf5
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    # 2 samples per file -> 50 files
    streamer = StreamToHDF5(data_folder_name, samples_per_file=2, keys=['input', 'support', 'target'])
    p.add_post_processor(streamer)
    state = p.execute()

    # 2. Load data from the SaveStateToList hook
    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['support']
    t_indices = state['data']['idx']['target']
    max_inp_len = np.max(state['data']['lengths']['input'])
    max_sup_len = np.max(state['data']['lengths']['support'])
    # For SNLI the targets consist of single words'
    assert np.max(state['data']['lengths']['target']) == 1, 'Max index label length should be 1'
    assert 'counts' in streamer.config, 'counts key not found in config dict!'
    assert len(streamer.config['counts']) > 0,'Counts of samples per file must be larger than zero (probably no files have been saved)'

    # 3. parse data to numpy
    n = len(inp_indices)
    X = np.zeros((n, max_inp_len), dtype=np.int64)
    X_len = np.zeros((n), dtype=np.int64)
    S = np.zeros((n, max_sup_len), dtype=np.int64)
    S_len = np.zeros((n), dtype=np.int64)
    t = np.zeros((n), dtype=np.int64)
    index = np.zeros((n), dtype=np.int64)

    for i in range(len(inp_indices)):
        sample_inp = inp_indices[i][0]
        sample_sup = sup_indices[i][0]
        sample_t = t_indices[i][0]
        l = len(sample_inp)
        X_len[i] = l
        X[i, :l] = sample_inp

        l = len(sample_sup)
        S_len[i] = l
        S[i, :l] = sample_sup

        t[i] = sample_t[0]
        index[i] = i

    # 4. setup expected paths
    inp_paths = [join(base_path, 'input_' + str(i) + '.hdf5') for i in range(1, 50)]
    sup_paths = [join(base_path, 'support_' + str(i) + '.hdf5') for i in range(1, 50)]
    target_paths = [join(base_path, 'target_' + str(i) + '.hdf5') for i in range(1, 50)]
    inp_len_paths = [join(base_path, 'input_lengths_' + str(i) + '.hdf5') for i in range(1, 50)]
    sup_len_paths = [join(base_path, 'support_lengths_' + str(i) + '.hdf5') for i in range(1, 50)]
    index_paths = [join(base_path, 'index_' + str(i) + '.hdf5') for i in range(1, 50)]

    data_idx = 0
    for path in index_paths:
        assert os.path.exists(path), 'Index path does not exist!'
        start = data_idx*2
        end = (data_idx + 1)*2
        data_idx += 1
        index[start:end] = load_hdf_file(path)

    X = X[index]
    S = S[index]
    t = t[index]
    X_len = X_len[index]
    S_len = S_len[index]
    zip_iter = zip([X, S, t, X_len, S_len], [inp_paths, sup_paths, target_paths, inp_len_paths, sup_len_paths ])
    print(index)

    # 5. Compare data
    for data, paths in zip_iter:
        data_idx = 0
        for path in paths:
            assert os.path.exists(path), 'This file should have been created by the HDF5Streamer: {0}'.format(path)
            shard = load_hdf_file(path)
            start = data_idx*2
            end = (data_idx + 1)*2
            np.testing.assert_array_equal(shard, data[start:end], 'HDF5 Stream data not equal for path {0}'.format(path))
            data_idx += 1

    # 6. compare config
    config_path = join(base_path, 'hdf5_config.pkl')
    config_reference = streamer.config
    assert os.path.exists(config_path), 'No HDF5 config exists under the path: {0}'.format(config_path)
    config_dict = pickle.load(open(config_path, 'rb'))
    assert 'paths' in config_dict, 'paths key not found in config dict!'
    assert 'fractions' in config_dict, 'fractions key not found in config dict!'
    assert 'counts' in config_dict, 'counts key not found in config dict!'
    for paths1, paths2 in zip(config_dict['paths'], streamer.config['paths']):
        for path1, path2 in zip(paths1, paths2):
            assert path1 == path2, 'Paths differ from HDF5 config!'
    np.testing.assert_array_equal(config_dict['fractions'], streamer.config['fractions'], 'Fractions for HDF5 samples per file not equal!')
    np.testing.assert_array_equal(config_dict['counts'], streamer.config['counts'], 'Counts for HDF5 samples per file not equal!')
    assert len(streamer.config['counts']) > 0, 'List of counts empty!'

    path_types = ['input', 'support', 'input_length', 'support_length', 'target', 'index']
    for i, paths in enumerate(streamer.config['paths']):
        assert len(paths) == 6, 'One path type is missing! Required path types {0}, existing paths {1}.'.format(path_types, paths)

    # 7. clean up
    shutil.rmtree(base_path)


def test_bin_search():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'snli3k_bins'
    total_samples = 30000.0
    base_path = join(get_data_path(), 'test_pipeline', data_folder_name)
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli3k'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveLengthsToState())
    p.execute()
    p.clear_processors()

    # 2. Execute the binning procedure
    p.add_path(get_test_data_path_dict()['snli3k'])
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    bin_creator = CreateBinsByNestedLength(data_folder_name, min_batch_size=4)
    p.add_post_processor(bin_creator)
    state = p.execute()

    # 3. We proceed to test if the bin sizes are correct, the config is correct, 
    #    if the calculated fraction of wasted samples is correct.
    #    This makes use of the state of the CreateBins class itself which
    #    thus biases this test. Use statiatical logging for 
    #    additional verification of correctness.

    # 3.1 Test config equality
    config_path = join(base_path, 'hdf5_config.pkl')
    assert os.path.exists(base_path), 'Base path for binning does not exist!'
    assert os.path.exists(config_path), 'Config file for binning not found!'
    config_dict = pickle.load(open(config_path, 'rb'))
    assert 'paths' in config_dict, 'paths key not found in config dict!'
    assert 'fractions' in config_dict, 'fractions key not found in config dict!'
    assert 'counts' in config_dict, 'counts key not found in config dict!'
    for paths1, paths2 in zip(config_dict['paths'], bin_creator.config['paths']):
        for path1, path2 in zip(paths1, paths2):
                assert path1 == path2, 'Paths differ from bin config!'
    np.testing.assert_array_equal(config_dict['fractions'], bin_creator.config['fractions'], 'Fractions for HDF5 samples per file not equal!')
    np.testing.assert_array_equal(config_dict['counts'], bin_creator.config['counts'], 'Counts for HDF5 samples per file not equal!')
    assert len(bin_creator.config['counts']) > 0, 'List of counts empty!'

    path_types = ['input', 'support', 'input_length', 'support_length', 'target', 'index']
    for i, paths in enumerate(bin_creator.config['paths']):
        assert len(paths) == 6, 'One path type is missing! Required path types {0}, existing paths {1}.'.format(path_types, paths)
    num_idxs = len(bin_creator.config['paths'])
    paths_inp = [join(base_path, 'input_bin_{0}.hdf5'.format(i)) for i in range(num_idxs)]
    paths_sup = [join(base_path, 'support_bin_{0}.hdf5'.format(i)) for i in range(num_idxs)]

    # 3.2 Test length, count and total count equality
    num_samples_bins = bin_creator.total_bin_count
    cumulative_count = 0.0
    for i, (path_inp, path_sup) in enumerate(zip(paths_inp, paths_sup)):
        inp = load_hdf_file(path_inp)
        sup = load_hdf_file(path_sup)
        l1 = bin_creator.config['path2len'][path_inp]
        l2 = bin_creator.config['path2len'][path_sup]
        count = bin_creator.config['path2count'][path_sup]

        expected_bin_fraction = count/num_samples_bins
        actual_bin_fraction = bin_creator.config['fractions'][i]

        assert actual_bin_fraction == expected_bin_fraction, 'Bin fraction for bin {0} not equal'.format(i)
        assert inp.shape[0] == count, 'Count for input bin at {0} not as expected'.format(path_inp)
        assert sup.shape[0] == count, 'Count for support bin at {0} not as expected'.format(path_sup)
        assert inp.shape[1] == l1, 'Input data sequence length for {0} not as expected'.format(path_inp)
        assert sup.shape[1] == l2, 'Support data sequence length for {0} not as expected'.format(path_sup)

        cumulative_count += count

    assert cumulative_count == num_samples_bins, 'Number of total bin samples not as expected!'

    shutil.rmtree(base_path)


batch_size = [17, 128]
samples_per_file = [500]
randomize = [True, False]
test_data = [r for r in itertools.product(samples_per_file, randomize, batch_size)]
test_data.append((1000000, True, 83))
str_func = lambda i, j, k: 'samples_per_file={0}, randomize={1}, batch_size={2}'.format(i, j, k)
ids = [str_func(i,j,k) for i,j,k in test_data]
test_idx = np.random.randint(0,len(test_data),3)
@pytest.mark.parametrize("samples_per_file, randomize, batch_size", test_data, ids=ids)
def test_non_random_stream_batcher(samples_per_file, randomize, batch_size):
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'snli_test'
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder, data_folder_name)
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline(pipeline_folder)
    p.add_path(get_test_data_path_dict()['snli1k'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveLengthsToState())
    p.execute()
    p.clear_processors()

    # 2. Process the data further to stream it to hdf5
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    # 2 samples per file -> 50 files
    streamer = StreamToHDF5(data_folder_name, samples_per_file=samples_per_file, keys=['input', 'support', 'target'])
    p.add_post_processor(streamer)
    state = p.execute()

    # 2. Load data from the SaveStateToList hook
    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['support']
    t_indices = state['data']['idx']['target']
    max_inp_len = np.max(state['data']['lengths']['input'])
    max_sup_len = np.max(state['data']['lengths']['support'])
    # For SNLI the targets consist of single words'
    assert np.max(state['data']['lengths']['target']) == 1, 'Max index label length should be 1'
    assert 'counts' in streamer.config, 'counts key not found in config dict!'
    assert len(streamer.config['counts']) > 0,'Counts of samples per file must be larger than zero (probably no files have been saved)'
    if samples_per_file == 100000:
        count = len(streamer.config['counts'])
        assert count == 1,'Samples per files is 100000 and there should be one file for 1k samples, but there are {0}'.format(count)

    assert streamer.num_samples == 1000, 'There should be 1000 samples for this dataset, but found {1}!'.format(streamer.num_samples)


    # 3. parse data to numpy
    n = len(inp_indices)
    X = np.zeros((n, max_inp_len), dtype=np.int64)
    X_len = np.zeros((n), dtype=np.int64)
    S = np.zeros((n, max_sup_len), dtype=np.int64)
    S_len = np.zeros((n), dtype=np.int64)
    T = np.zeros((n), dtype=np.int64)

    for i in range(len(inp_indices)):
        sample_inp = inp_indices[i][0]
        sample_sup = sup_indices[i][0]
        sample_t = t_indices[i][0]
        l = len(sample_inp)
        X_len[i] = l
        X[i, :l] = sample_inp

        l = len(sample_sup)
        S_len[i] = l
        S[i, :l] = sample_sup

        T[i] = sample_t[0]

    epochs = 2
    batcher = StreamBatcher(pipeline_folder, data_folder_name, batch_size, loader_threads=8, randomize=randomize)
    del batcher.at_batch_prepared_observers[:]

    # 4. test data equality
    for epoch in range(epochs):
        for x, x_len, s, s_len, t, idx in batcher:
            assert np.int32 == x_len.dtype, 'Input length type should be int32!'
            assert np.int32 == s_len.dtype, 'Support length type should be int32!'
            assert np.int32 == x.dtype, 'Input type should be int32!'
            assert np.int32 == s.dtype, 'Input type should be int32!'
            assert np.int32 == t.dtype, 'Target type should be int32!'
            assert np.int32 == idx.dtype, 'Index type should be int32!'
            np.testing.assert_array_equal(X[idx], x, 'Input data not equal!')
            np.testing.assert_array_equal(S[idx], s, 'Support data not equal!')
            np.testing.assert_array_equal(X_len[idx], x_len, 'Input length data not equal!')
            np.testing.assert_array_equal(S_len[idx], s_len, 'Support length data not equal!')
            np.testing.assert_array_equal(T[idx], t, 'Target data not equal!')

    # 5. clean up
    shutil.rmtree(base_path)

def test_abitrary_input_data():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    base_path = join(get_data_path(), 'test_keys')
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    file_path = join(get_data_path(), 'test_pipeline', 'test_data.json')

    questions = [['bla bla Q1', 'this is q2', 'q3'], ['q4 set2', 'or is it q1?']]
    support = [['I like', 'multiple supports'], ['yep', 'they are pretty cool', 'yeah, right?']]
    answer = [['yes', 'absolutly', 'not really'], ['you bet', 'yes']]
    pos_tag = [['t1', 't2'], ['t1', 't2', 't3']]

    with open(file_path, 'w') as f:
        for i in range(2):
            f.write(json.dumps([questions[i], support[i], answer[i], pos_tag[i]]) + '\n')

    keys2keys = {}
    keys2keys['answer'] = 'answer'
    keys2keys['pos'] = 'pos'
    p = Pipeline('test_keys', keys=['question', 'support', 'answer', 'pos'])
    p.add_path(file_path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab(general_vocab_keys=['question', 'support']))
    p.add_post_processor(SaveLengthsToState())
    p.execute()

    p.clear_processors()
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(ConvertTokenToIdx(keys2keys=keys2keys))
    p.add_post_processor(StreamToHDF5('test', keys=['question', 'support', 'answer', 'pos']))
    p.add_post_processor(SaveStateToList('data'))
    state = p.execute()

    Q = state['data']['data']['question']
    S = state['data']['data']['support']
    A = state['data']['data']['answer']
    pos = state['data']['data']['pos']
    #vocab is offset by 2, due to UNK and empty word ''
    # note that we travers the data like q1, s1, a1; q2, s2, a2
    # we share vocab between question and support
    expected_Q_ids = [[[ 2, 2, 3], [4, 5, 6], [7]], [[12, 13], [14, 5, 15, 16, 17]]]
    expected_S_ids = [[[8, 9], [10, 11]], [[18], [19, 20, 21, 22], [23, 24, 25, 17]]]
    expected_answer_ids = [[[2],[3],[4, 5]],[[6,7], [2]]]
    expected_pos_ids = [[[2],[3]],[[2],[3],[4]]]

    np.testing.assert_array_equal(np.array(expected_Q_ids), Q)
    np.testing.assert_array_equal(np.array(expected_S_ids), S)
    np.testing.assert_array_equal(np.array(expected_answer_ids), A)
    np.testing.assert_array_equal(np.array(expected_pos_ids), pos)


def test_bin_streamer():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'bin_snli_test'
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder, data_folder_name)
    batch_size = 4
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline(pipeline_folder)
    p.add_path(get_test_data_path_dict()['snli1k'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveLengthsToState())
    p.execute()
    p.clear_processors()

    # 2. Process the data further to stream it to hdf5
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    # 2 samples per file -> 50 files
    bin_creator = CreateBinsByNestedLength(data_folder_name, min_batch_size=batch_size, raise_on_throw_away_fraction=0.5)
    p.add_post_processor(bin_creator)
    state = p.execute()

    # 2. Load data from the SaveStateToList hook
    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['support']
    t_indices = state['data']['idx']['target']
    max_inp_len = np.max(state['data']['lengths']['input'])
    max_sup_len = np.max(state['data']['lengths']['support'])
    # For SNLI the targets consist of single words'
    assert np.max(state['data']['lengths']['target']) == 1, 'Max index label length should be 1'

    # 3. parse data to numpy
    n = len(inp_indices)
    X = np.zeros((n, max_inp_len), dtype=np.int64)
    X_len = np.zeros((n), dtype=np.int64)
    S = np.zeros((n, max_sup_len), dtype=np.int64)
    S_len = np.zeros((n), dtype=np.int64)
    T = np.zeros((n), dtype=np.int64)

    for i in range(len(inp_indices)):
        sample_inp = inp_indices[i][0]
        sample_sup = sup_indices[i][0]
        sample_t = t_indices[i][0]
        l = len(sample_inp)
        X_len[i] = l
        X[i, :l] = sample_inp

        l = len(sample_sup)
        S_len[i] = l
        S[i, :l] = sample_sup

        T[i] = sample_t[0]

    epochs = 3
    batcher = StreamBatcher(pipeline_folder, data_folder_name, batch_size, loader_threads=8, randomize=True)
    del batcher.at_batch_prepared_observers[:] # we want to test on raw numpy data

    # 4. test data equality
    for epoch in range(epochs):
        for x, x_len, s, s_len, t, idx in batcher:
            assert np.int32 == x_len.dtype, 'Input length type should be int32!'
            assert np.int32 == s_len.dtype, 'Support length type should be int32!'
            assert np.int32 == x.dtype, 'Input type should be int32!'
            assert np.int32 == s.dtype, 'Input type should be int32!'
            assert np.int32 == t.dtype, 'Target type should be int32!'
            assert np.int32 == idx.dtype, 'Index type should be int32!'
            np.testing.assert_array_equal(X[idx, :x_len[0]], x, 'Input data not equal!')
            np.testing.assert_array_equal(S[idx, :s_len[0]], s, 'Support data not equal!')
            np.testing.assert_array_equal(X_len[idx], x_len, 'Input length data not equal!')
            np.testing.assert_array_equal(S_len[idx], s_len, 'Support length data not equal!')
            np.testing.assert_array_equal(T[idx], t, 'Target data not equal!')

            # if the next tests fail, it means the batches provides the wrong length for the sample
            np.testing.assert_array_equal(S[idx, s_len[0]:], np.zeros((batch_size, S.shape[1]-s_len[0])), 'Support tail not padded exclusively with zeros!')
            np.testing.assert_array_equal(X[idx, x_len[0]:], np.zeros((batch_size, X.shape[1]-x_len[0])), 'Input tail not padded exclusively with zeros!')

    # 5. clean up
    shutil.rmtree(base_path)


names = ['loss', 'accuracy']
print_every = [20, 7, 13, 2000]
test_data = [r for r in itertools.product(names, print_every)]
ids = ['name={0}, print_every={1}'.format(name, print_every) for name, print_every in test_data]
@pytest.mark.parametrize("hook_name, print_every", test_data, ids=ids)
def test_hook(hook_name, print_every):
    def calc_confidence_interval(expected_loss):
        mean = np.mean(expected_loss)
        std = np.std(expected_loss)
        z = scipy.stats.norm.ppf(0.99)
        se = z*std/np.sqrt(print_every)
        lower_expected = mean-se
        upper_expected = mean+se
        return lower_expected, upper_expected, mean, n

    def generate_loss():
        loss = np.random.rand()
        state = BatcherState()
        state.loss = loss
        return loss, state

    def generate_accuracy():
        target = np.random.randint(0,3,print_every)
        argmax = np.random.randint(0,3,print_every)
        state = BatcherState()
        state.targets = target
        state.argmax = argmax
        accuracy = np.mean(target==argmax)
        return accuracy, state

    if hook_name == 'loss':
        hook = LossHook(print_every_x_batches=print_every)
        gen_func = generate_loss
    elif hook_name == 'accuracy':
        hook = AccuracyHook(print_every_x_batches=print_every)
        gen_func = generate_accuracy

    expected_loss = []
    state = BatcherState()
    for epoch in range(2):
        for i in range(100):
            metric, state = gen_func()
            expected_loss.append(metric)
            lower, upper, m, n = hook.at_end_of_iter_event(state)
            if (i+1) % print_every == 0:
                lower_expected, upper_expected, mean, n2 = calc_confidence_interval(expected_loss)
                print(i, epoch)
                assert n == n2, 'Sample size not equal!'
                assert np.allclose(m, mean), 'Mean not equal!'
                assert np.allclose(lower, lower_expected), 'Lower confidence bound not equal!'
                assert np.allclose(upper, upper_expected), 'Upper confidence bound not equal!'
                del expected_loss[:]

        lower, upper, m, n = hook.at_end_of_epoch_event(state)
        lower_expected, upper_expected, mean, n2 = calc_confidence_interval(expected_loss)
        del expected_loss[:]


def test_variable_duplication():
    nltk.download('averaged_perceptron_tagger')
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder)
    batch_size = 32
    func = lambda x: [tag for word, tag in nltk.pos_tag(x)]
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    # 1. Setup pipeline to save lengths and generate vocabulary
    keys = ['input', 'support', 'target', 'input_pos']
    keys2keys = { 'input_pos' : 'input'}
    p = Pipeline(pipeline_folder, keys=keys, keys2keys=keys2keys)
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_sent_processor(SaveStateToList('tokens'))
    p.add_post_processor(SaveLengthsToState())
    p.add_post_processor(DeepSeqMap(func), keys=['input_pos'])
    p.execute()
    p.clear_processors()

    # 2. Process the data further to stream it to hdf5
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(DeepSeqMap(func), keys=['input_pos'])
    p.add_post_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx(keys2keys={'input_pos' : 'input_pos'}))
    p.add_post_processor(SaveStateToList('idx'))
    # 2 samples per file -> 50 files
    #p.add_post_processor(StreamToHDF5(data_folder_name, keys=keys))
    state = p.execute()

    # 2. Load data from the SaveStateToList hook
    inp_sents = state['data']['tokens']['input']
    pos_tags = state['data']['idx']['input_pos']
    vocab = p.state['vocab']['input_pos']

    print(vocab.idx2token)
    print(pos_tags[0][0])


    tags_expected = []
    for sent in inp_sents:
        tag_tuples = nltk.pos_tag(sent)
        tag = [tag for word, tag in tag_tuples]
        tags_expected.append(tag)

    tags = []
    for sent in pos_tags[0]:
        tag = [vocab.get_word(idx) for idx in sent]
        tags.append(tag)


    print(tags[0])
    print(tags_expected[0])
    for tags1, tags2 in zip(tags, tags_expected):
        assert len(tags1) == len(tags2), 'POS tag lengths not the same!'
        for tag1, tag2 in zip(tags1, tags2):
            assert tag1 == tag2, 'POS tags were not the same!'

    # 5. clean up
    shutil.rmtree(base_path)
