import sys
import json
import re
from time import sleep
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

def read_data(data_filename):
    with open(data_filename) as data_file:
        data = json.load(data_file)
        return data


#for i in range(21):
    # the exact output you're looking for:
#    sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
#    sys.stdout.flush()
#    sleep(0.25)

def annotate_corpus(data):
    total = len(data)
    partitions = 50
    partition_size = total / partitions
    inc = 0
    annotated = []
    for i, instance in enumerate(data):
#        annotated.append(annotate_instance(instance))
        if i / partition_size > inc+1 or i == total-1:
            inc += 1
            sys.stdout.write("\r[" + ("=" * inc) +  " " * (partitions - inc) + "]" +  str(inc*2) + "%")
            sys.stderr.flush()
    return annotated
#    return [annotate_instance(instance) for instance in data]

def annotate_instance(instance):
    # weird mix of mutable + returns here, but okay for now
    instance['support'] = [annotate_support(support) for support in instance['support']]
    return instance

def annotate_support(support):
    support_text = support['text']
    annotations = annotate_text(support_text)
    try:
        annotations.keys()
    except:
        annotations = json.loads(annotations, encoding='utf-8', strict=False)
    token_offsets = []
    sentence_offsets = []
    postags = []
    parses = []
    try:
        for sentence in annotations['sentences']:
            for token in sentence['tokens']:
                token_offsets.append([token['characterOffsetBegin'], token['characterOffsetEnd']])
                postags.append(token['pos'])
            sentence_offsets.append([sentence['tokens'][0]['characterOffsetBegin'], sentence['tokens'][-1]['characterOffsetEnd']])
            parses.append(clean_parse(sentence['parse']))
    except:
        print('Error annotating text: \n', support_text)
        sys.exit
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
    ann_str = 'tokenize,ssplit,pos,parse' #,depparse,
    output = nlp.annotate(text,
        properties={
            'annotators': ann_str,
            'timeout': '50000',
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
