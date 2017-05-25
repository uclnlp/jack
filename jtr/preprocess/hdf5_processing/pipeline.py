from os.path import join

import os
import shutil
import simplejson as json
import zipfile

from spodernet.preprocessing.vocab import Vocab
from spodernet.utils.util import Timer

from spodernet.utils.logger import Logger
log = Logger('pipeline.py.txt')

t = Timer()

class Pipeline(object):
    def __init__(self, name, delete_all_previous_data=False, keys=None, keys2keys=None):
        self.line_processors = []
        self.text_processors = []
        self.sent_processors = []
        self.token_processors = []
        self.post_processors = []
        self.paths = []

        self.keys = keys or ['input', 'support', 'target']
        if keys2keys is None:
            keys2keys = {}
            for key in self.keys:
                keys2keys[key] = key
        else:
            for key in self.keys:
                if key not in keys2keys:
                    keys2keys[key] = key

        self.keys2keys = keys2keys
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
        for key in self.keys:
            self.state['vocab'][key] = Vocab(path=join(self.root, 'vocab_'+key))

    def add_line_processor(self, line_processor):
        self.line_processors.append(line_processor)

    def add_text_processor(self, text_processor, keys=None):
        keys = keys or self.keys
        text_processor.link_with_pipeline(self.state)
        log.debug('Added text preprocessor {0}', type(text_processor))
        self.text_processors.append([keys, text_processor])

    def add_sent_processor(self, sent_processor, keys=None):
        keys = keys or self.keys
        sent_processor.link_with_pipeline(self.state)
        log.debug('Added sent preprocessor {0}', type(sent_processor))
        self.sent_processors.append([keys, sent_processor])

    def add_token_processor(self, token_processor, keys=None):
        keys = keys or self.keys
        token_processor.link_with_pipeline(self.state)
        log.debug('Added token preprocessor {0}', type(token_processor))
        self.token_processors.append([keys, token_processor])

    def add_post_processor(self, post_processor, keys=None):
        keys = keys or self.keys
        post_processor.link_with_pipeline(self.state)
        log.debug('Added post preprocessor {0}', type(post_processor))
        self.post_processors.append([keys, post_processor])

    def add_path(self, path):
        log.debug('Added path to JSON file {0}', path)
        self.paths.append(path)

    def stream_file(self, path):
        log.debug('Processing file {0}'.format(path))
        file_handle = None
        key2idx = {}
        for i, key in enumerate(self.keys):
            if key not in key2idx:
                key2idx[key] = i

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
                data = []
                for key in self.keys:
                    data_key = self.keys2keys[key]
                    idx = key2idx[data_key]
                    data.append(line[idx])


                yield data

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
        for key in self.keys:
            self.state['vocab'][key].save_to_disk()

    def load_vocabs(self):
        self.state['vocab']['general'].load_from_disk()
        for key in self.keys:
            self.state['vocab'][key].load_from_disk()

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

    def iterate_over_processors(self, processors, variables):
        for filter_keys, textp in processors:
            for i, key in enumerate(self.keys):
                if key in filter_keys:
                    variables[i] = textp.process(variables[i], inp_type=key)
        return variables

    def execute(self):
        '''Tokenizes the data, calcs the max length, and creates a vocab.'''

        for path in self.paths:
            for var in self.stream_file(path):
                t.tick('text')
                for filter_keys, textp in self.text_processors:
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            var[i] = textp.process(var[i], inp_type=key)
                t.tick('text')

                for i in range(len(var)):
                    var[i] = (var[i] if isinstance(var[i], list) else [var[i]])

                t.tick('sent')
                for filter_keys, sentp in self.sent_processors:
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            for j in range(len(var[i])):
                                var[i][j] = sentp.process(var[i][j], inp_type=key)
                t.tick('sent')

                for i in range(len(var)):
                    var[i] = (var[i] if isinstance(var[i][0], list) else [[sent] for sent in var[i]])

                t.tick('token')
                for filter_keys, tokenp in self.token_processors:
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            for j in range(len(var[i])):
                                for k in range(len(var[i][j])):
                                    var[i][j][k] = tokenp.process(var[i][j][k], inp_type=key)
                t.tick('token')

                t.tick('post')
                for filter_keys, postp in self.post_processors:
                    for i, key in enumerate(self.keys):
                        if key in filter_keys:
                            var[i] = postp.process(var[i], inp_type=key)
                t.tick('post')

        t.tick('postpost')
        #for key in self.keys:
        #    for keys, textp in self.text_processors: textp.post_process(key)
        #    for keys, sentp in self.sent_processors: sentp.post_process(key)
        #    for keys, tokenp in self.token_processors: tokenp.post_process(key)
        #    for keys, postp in self.post_processors: postp.post_process(key)
        t.tock('postpost')
        t.tock('text')
        t.tock('sent')
        t.tock('token')
        t.tock('post')
        return self.state


    
