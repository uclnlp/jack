import json

def read_data(data_filename):
    with open(data_filename) as data_file:
        data = json.load(data_file)
        return data

def read_rich_data(data_filename):
    return [RichInstance(x) for x in read_data(data_filename)]

class RichInstance(object):
    def __init__(self, instance):
        self.instance = instance

    def vocab(self):
        return self.question_vocab().union(self.support_vocab())

    def question_vocab(self):
        tokens = set()
        for question in self.instance['questions']:
            for token in question['question'].split(' '):
                tokens.add(token)
        return tokens

    def support_vocab(self):
        tokens = set()
        for support in self.instance['support']:
            for token in support['text'].split(' '):
                tokens.add(token)
        return tokens
