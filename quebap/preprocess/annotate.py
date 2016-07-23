import sys
import json
import re
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

def read_data(data_filename):
    with open(data_filename) as data_file:
        data = json.load(data_file)
        return data

def annotate_corpus(data):
    return [annotate_instance(instance) for instance in data]

def annotate_instance(instance):
    # weird mix of mutable + returns here, but okay for now
    instance['support'] = [annotate_support(support) for support in instance['support']]
    return instance

def annotate_support(support):
    support_text = support['text']
    annotations = annotate_text(support_text)
    token_offsets = []
    sentence_offsets = []
    postags = []
    parses = []
    for sentence in annotations['sentences']:
        for token in sentence['tokens']:
            token_offsets.append([token['characterOffsetBegin'], token['characterOffsetEnd']])
            postags.append(token['pos'])
        sentence_offsets.append([sentence['tokens'][0]['characterOffsetBegin'], sentence['tokens'][-1]['characterOffsetEnd']])
        parses.append(clean_parse(sentence['parse']))
#    return_dict = {'text': support_text}
    return {
        'text': support_text,
        'tokens': token_offsets,
        'sentences': sentence_offsets,
        'postags': postags,
        'parses': parses
    }

def annotate_text(text):
    text = clean_for_annotator(text)
    ann_str = 'tokenize,ssplit,pos,depparse,parse'
    output = nlp.annotate(text,
        properties={
            'annotators': ann_str,
            'outputFormat': 'json'})
    return output

def clean_for_annotator(text):
    text = text.replace('–', '-')
    return text

def clean_parse(parse):
    parse_str = str(parse)
    parse_str = parse_str.replace('\n', ' ')
    parse_str = re.sub(r' +', ' ', parse_str)
    return parse_str

def main():
    import sys
    if len(sys.argv) == 2:
        data = annotate_corpus(read_data(sys.argv[1]))
        print(json.dumps(data, indent=2))

if __name__ == "__main__": main()
