from os.path import join

import os
import shutil
import json
import zipfile

from jtr.preprocess.hdf5_processing.vocab import Vocab

from jtr.util.logger import Logger
log = Logger('pipeline.py.txt')

class Pipeline(object):
    def __init__(self, name, delete_all_previous_data=False):
        self.line_processors = []
        self.text_processors = []
        self.sent_processors = []
        self.token_processors = []
        self.post_processors = []
        self.paths = []
        home = os.environ['HOME']
        self.root = join(home, '.data', name)
        if not os.path.exists(self.root):
            log.debug_once('Pipeline path {0} does not exist. Creating folder...', self.root)
            os.mkdir(self.root)
        else:
            if delete_all_previous_data:
                log.warning('delete_all_previous_data=True! Deleting all folder contents of folder {0}!', self.root)
                shutil.rmtree(self.root)
                log.info('Recreating path: {0}', self.root)
                os.mkdir(self.root)
            else:
                log.warning('Pipeline path {0} already exist. This pipeline may overwrite data in this path!', self.root)

        self.state = {'name' : name, 'home' : home, 'path' : self.root, 'data' : {}}
        self.state['vocab'] = {}
        self.state['vocab']['general'] = Vocab(path=join(self.root, 'vocab'))
        self.state['vocab']['input'] = Vocab(path=join(self.root, 'vocab_input'))
        self.state['vocab']['support'] = Vocab(path=join(self.root, 'vocab_support'))
        self.state['vocab']['target'] = Vocab(path=join(self.root, 'vocab_target'))

    def add_line_processor(self, line_processor):
        self.line_processors.append(line_processor)

    def add_text_processor(self, text_processor, keys=['input', 'support', 'target']):
        text_processor.link_with_pipeline(self.state)
        log.debug('Added text preprocessor {0}', type(text_processor))
        self.text_processors.append([keys, text_processor])

    def add_sent_processor(self, sent_processor, keys=['input', 'support', 'target']):
        sent_processor.link_with_pipeline(self.state)
        log.debug('Added sent preprocessor {0}', type(sent_processor))
        self.sent_processors.append([keys, sent_processor])

    def add_token_processor(self, token_processor, keys=['input', 'support', 'target']):
        token_processor.link_with_pipeline(self.state)
        log.debug('Added token preprocessor {0}', type(token_processor))
        self.token_processors.append([keys, token_processor])

    def add_post_processor(self, post_processor, keys=['input', 'support', 'target']):
        post_processor.link_with_pipeline(self.state)
        log.debug('Added post preprocessor {0}', type(post_processor))
        self.post_processors.append([keys, post_processor])

    def add_path(self, path):
        log.debug('Added path to JSON file {0}', path)
        self.paths.append(path)

    def stream_file(self, path):
        log.debug('Processing file {0}'.format(path))
        file_handle = None
        if '.zip' in path:
            path_to_zip, path_to_file = path.split('.zip')
            path_to_zip += '.zip'
            path_to_file = path_to_file[1:]

            archive = zipfile.ZipFile(path_to_zip, 'r')
            file_handle = archive.open(path_to_file, 'r')
        else:
            file_handle = open(path)

        for line in file_handle:
            filtered = False
            for linep in self.line_processors:
                line = linep.process(line)
                if line is None:
                    filtered = True
                    break
            if filtered:
                continue
            else:
                log.debug_once('First line processed by line processors: {0}', line)
                yield line

    def clear_processors(self):
        self.post_processors = []
        self.sent_processors = []
        self.token_processors = []
        self.text_processors = []
        log.debug('Cleared processors of pipeline {0}', self.state['name'])

    def clear_paths(self):
        self.paths = []

    def save_vocabs(self):
        self.state['vocab']['general'].save_to_disk()
        self.state['vocab']['input'].save_to_disk()
        self.state['vocab']['support'].save_to_disk()
        self.state['vocab']['target'].save_to_disk()

    def load_vocabs(self):
        self.state['vocab']['general'].load_from_disk()
        self.state['vocab']['input'].load_from_disk()
        self.state['vocab']['support'].load_from_disk()
        self.state['vocab']['target'].load_from_disk()

    def copy_vocab_from_pipeline(self, pipeline_or_vocab, vocab_type=None):
        if isinstance(pipeline_or_vocab, Pipeline):
            self.state['vocab'] = pipeline_or_vocab.state['vocab']
        elif isinstance(pipeline_or_vocab, Vocab):
            if vocab_type is None:
                self.state['vocab']['general'] = pipeline_or_vocab
            else:
                self.state['vocab'][vocab_type] = pipeline_or_vocab
        else:
            str_error = 'The add vocab method expects a Pipeline or Vocab instance as argument, got {0} instead!'.format(type(pipeline_or_vocab))
            log.error(str_error)
            raise TypeError(str_error)

    def execute(self):
        '''Tokenizes the data, calcs the max length, and creates a vocab.'''

        for path in self.paths:
            for inp, sup, target in self.stream_file(path):
                for keys, textp in self.text_processors:
                    if 'input' in keys:
                        inp = textp.process(inp, inp_type='input')
                    if 'support' in keys:
                        sup = textp.process(sup, inp_type='support')
                    if 'target' in keys:
                        target = textp.process(target, inp_type='target')

                inp_sents = (inp if isinstance(inp, list) else [inp])
                sup_sents = (sup if isinstance(sup, list) else [sup])
                t_sents = (target if isinstance(target, list) else [target])

                for keys, sentp in self.sent_processors:
                    if 'input' in keys:
                        for i in range(len(inp_sents)):
                            inp_sents[i] = sentp.process(inp_sents[i], inp_type='input')
                    if 'support' in keys:
                        for i in range(len(sup_sents)):
                            sup_sents[i] = sentp.process(sup_sents[i], inp_type='support')

                    if 'target' in keys:
                        for i in range(len(t_sents)):
                            t_sents[i] = sentp.process(t_sents[i], inp_type='target')

                inp_words = (inp_sents if isinstance(inp_sents[0], list) else [[sent] for sent in inp_sents])
                sup_words = (sup_sents if isinstance(sup_sents[0], list) else [[sent] for sent in sup_sents])
                t_words = (t_sents if isinstance(t_sents[0], list) else [[sent] for sent in t_sents])

                for keys, tokenp in self.token_processors:
                    if 'input' in keys:
                        for i in range(len(inp_words)):
                            for j in range(len(inp_words[i])):
                                inp_words[i][j] = tokenp.process(inp_words[i][j], inp_type='input')

                    if 'support' in keys:
                        for i in range(len(sup_words)):
                            for j in range(len(sup_words[i])):
                                sup_words[i][j] = tokenp.process(sup_words[i][j], inp_type='support')

                    if 'target' in keys:
                        for i in range(len(t_words)):
                            for j in range(len(t_words[i])):
                                t_words[i][j] = tokenp.process(t_words[i][j], inp_type='target')

                for keys, postp in self.post_processors:
                    if 'input' in keys:
                        inp_words = postp.process(inp_words, inp_type='input')
                    if 'support' in keys:
                        sup_words = postp.process(sup_words, inp_type='support')
                    if 'target' in keys:
                        t_words = postp.process(t_words, inp_type='target')

        return self.state
