import os
import shutil
import zipfile
from os.path import join

from jack.util.hdf5_processing.vocab import Vocab
from jack.util.util import Timer
from jack.data_structures import QASetting, Answer
from jack.util.batch import GeneratorWithRestart

t = Timer()

class DatasetStreamer(object):
    def __init__(self, input_keys=None, output_keys=None):
        self.stream_processors = []
        self.input_keys = input_keys or ['input', 'support', 'target']
        self.output_keys = output_keys
        self.paths = []
        self.output_keys = output_keys or self.input_keys

    def add_stream_processor(self, streamp):
        self.stream_processors.append(streamp)

    def set_paths(self, list_of_paths):
        self.paths = list_of_paths

    def set_path(self, path):
        self.set_paths([path])

    def __iter__(self):
        return self

    def stream(self):
        for path in self.paths:
            with open(path) as f:
                for i, line in enumerate(f):
                    filtered = False
                    for streamp in self.stream_processors:
                        line = streamp.process(line)
                        if line is None:
                            filtered = True
                            break
                    if filtered:
                        continue
                    else:
                        data = []
                        inputkey2data = {}
                        for input_key, variable in zip(self.input_keys, line):
                            inputkey2data[input_key] = variable

                        for output_key in self.output_keys:
                            data.append(inputkey2data[output_key])


                        qa = QASetting(question=data[0], support=[data[1]], id=str(i),atomic_candidates=None)
                        a = Answer(data[2])

                        yield qa, a

class Pipeline(object):
    def __init__(self, name, delete_all_previous_data=False, keys=None):
        self.text_processors = []
        self.sent_processors = []
        self.token_processors = []
        self.post_processors = []

        self.keys = keys or ['input', 'support', 'target']
        home = os.environ['HOME']
        self.root = join(home, '.data', name)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        else:
            if delete_all_previous_data:
                shutil.rmtree(self.root)
                os.mkdir(self.root)
            else:
                pass

        self.state = {'name' : name, 'home' : home, 'path' : self.root, 'data' : {}}
        self.state['vocab'] = {}
        self.state['vocab']['general'] = Vocab(path=join(self.root, 'vocab'))
        for key in self.keys:
            self.state['vocab'][key] = Vocab(path=join(self.root, 'vocab_'+key))

    def add_text_processor(self, text_processor, keys=None):
        keys = keys or self.keys
        text_processor.link_with_pipeline(self.state)
        self.text_processors.append([keys, text_processor])

    def add_sent_processor(self, sent_processor, keys=None):
        keys = keys or self.keys
        sent_processor.link_with_pipeline(self.state)
        self.sent_processors.append([keys, sent_processor])

    def add_token_processor(self, token_processor, keys=None):
        keys = keys or self.keys
        token_processor.link_with_pipeline(self.state)
        self.token_processors.append([keys, token_processor])

    def add_post_processor(self, post_processor, keys=None):
        keys = keys or self.keys
        post_processor.link_with_pipeline(self.state)
        self.post_processors.append([keys, post_processor])


    def clear_processors(self):
        self.post_processors = []
        self.sent_processors = []
        self.token_processors = []
        self.text_processors = []

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
            raise TypeError(str_error)

    def iterate_over_processors(self, processors, variables):
        for filter_keys, textp in processors:
            for i, key in enumerate(self.keys):
                if key in filter_keys:
                    variables[i] = textp.process(variables[i], inp_type=key)
        return variables

    def execute(self, generator):
        '''Tokenizes the data, calcs the max length, and creates a vocab.'''
        for qa_settings, answer in generator:
            var = [qa_settings.question, qa_settings.support[0], answer.text]
            for filter_keys, textp in self.text_processors:
                for i, key in enumerate(self.keys):
                    if key in filter_keys:
                        var[i] = textp.process(var[i], inp_type=key)

            for i in range(len(var)):
                var[i] = (var[i] if isinstance(var[i], list) else [var[i]])

            for filter_keys, sentp in self.sent_processors:
                for i, key in enumerate(self.keys):
                    if key in filter_keys:
                        for j in range(len(var[i])):
                            var[i][j] = sentp.process(var[i][j], inp_type=key)

            for i in range(len(var)):
                var[i] = (var[i] if isinstance(var[i][0], list) else [[sent] for sent in var[i]])

            for filter_keys, tokenp in self.token_processors:
                for i, key in enumerate(self.keys):
                    if key in filter_keys:
                        for j in range(len(var[i])):
                            for k in range(len(var[i][j])):
                                var[i][j][k] = tokenp.process(var[i][j][k], inp_type=key)

            for filter_keys, postp in self.post_processors:
                for i, key in enumerate(self.keys):
                    if key in filter_keys:
                        var[i] = postp.process(var[i], inp_type=key)

        return self.state
