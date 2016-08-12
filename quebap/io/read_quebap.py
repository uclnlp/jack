import json
import re
token_pattern = re.compile('[^ ]+')

def read_data(data_filename):
    with open(data_filename) as data_file:
        data = json.load(data_file)
        return data

def read_rich_data(data_filename):
    return [RichInstance(x) for x in read_data(data_filename)]

class RichInstance(object):
    def __init__(self, instance):
        self.instance = instance
        self.supports = [Support(x) for x in self.instance['support']]
        self.questions = [Question(x) for x in self.instance['questions']]

    def vocab(self):
        tokens = set()
        for question in self.questions:
            for token in question.tokens:
                tokens.add(token)
        for support in self.supports:
            for token in support.tokens:
                tokens.add(token)
        return tokens

    def question_support_pairs(self):
        for q in self.questions:
            for s in self.supports:
                yield q,s

class Question(object):

    def __init__(self, qdict):
        self.source = qdict
        self.text = qdict['question']
        self.answers = [Answer(x) for x in qdict['answers']]
        self.tokens = self.text.split(' ')

class Answer(object):
    def __init__(self, adict):
        self.adict = adict
        self.text = adict['text']
        self.span = adict['span']

class Support(object):

    def __init__(self, sdict):
        self.sdict = sdict
        self.text = sdict['text']
        if 'tokens' in sdict:
            self.token_offsets = sdict['tokens']
        else:
            self.token_offsets = [m.span() for m in token_pattern.finditer(s)]
        self.tokens = [self.text[span[0]:span[1]] for span in self.token_offsets]

    def token_from_char(self, char_offset):
        for i,t in enumerate(self.token_offsets):
            if char_offset <= t[1]:
                return i
        return -1
