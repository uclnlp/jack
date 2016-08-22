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

def annotate_corpus(corpus, tokenize=False, sent_split=False, postag=False, parse=False, dep_parse=False):
    data = corpus['data']
    total = len(data)
    partitions = 50
    partition_size = total / partitions
    inc = 0
    annotated = []
    ann_str = core_nlp_annotations_str(tokenize, sent_split, postag, parse, dep_parse)
    l = len(ann_str)
    for i, instance in enumerate(data):
# weird mix of mutable + returns here, but okay for now
        instance['support'] = [annotate_text(support['text'],
                                                  tokenize=tokenize,
                                                  sent_split=sent_split,
                                                  postag=postag,
                                                  parse=parse,
                                                  dep_parse=dep_parse) for support in instance['support']]
        instance['questions'] = [{'question': annotate_text(question['question']['text'],
                                                  tokenize=tokenize,
                                                  sent_split=sent_split,
                                                  postag=postag,
                                                  parse=parse,
                                                  dep_parse=dep_parse)} for question in instance['questions']]
        annotated.append(instance)
        if i / partition_size > inc+1 or i == total-1:
            inc += 1
            sys.stderr.write("\r[" + ("=" * inc) +  " " * (partitions - inc) + "]" +  str(inc*2) + "%")
            sys.stderr.flush()
    return {
        'meta': {
            'source': corpus['meta']['source'],
            'annotations': 'core_nlp:' + core_nlp_annotations_str(tokenize, sent_split, postag, parse, dep_parse)
        },
        'data': annotated
    }

def annotate_text(text, tokenize=False, sent_split=False, postag=False, parse=False, dep_parse=False):
    annotations = annotate_with_corenlp(text, tokenize=tokenize,
                                              sent_split=sent_split,
                                              postag=postag,
                                              parse=parse,
                                              dep_parse=dep_parse)
    try:
        annotations.keys()
    except:
        annotations = json.loads(annotations, encoding='utf-8', strict=False)
    token_offsets = []
    tokens = []
    sentence_offsets = []
    postags = []
    parses = []
    dep_parses = []
    try:
        for sentence in annotations['sentences']:
            if tokenize:
                for token in sentence['tokens']:
                    token_offsets.append([token['characterOffsetBegin'], token['characterOffsetEnd']])
                    tokens.append(token['word'])
                    if 'pos' in token:
                        postags.append(token['pos'])
            if sent_split:
                sentence_offsets.append([sentence['tokens'][0]['characterOffsetBegin'], sentence['tokens'][-1]['characterOffsetEnd']])
            if 'parse' in sentence:
                parses.append(clean_parse(sentence['parse']))
            if 'depparse' in sentence:
                dep_parses.append(sentence['depparse'])
    except:
        print('Error annotating text: \n', text)
        sys.exit
    result = {}
    result['text'] = text
    if tokenize:
        result['tokens'] = tokens
        result['token_offsets'] = token_offsets
    if sent_split:
        result['sentences'] = sentence_offsets
    if postag:
        result['postags'] = postags
    if parse:
        result['parses'] = parses
    if dep_parse:
        result['dep_parses'] = dep_parses
    return result

def annotate_with_corenlp(text, tokenize=False, sent_split=False, postag=False, parse=False, dep_parse=False):
    text = text #clean_for_annotator(text)
    ann_str = 'tokenize,ssplit,pos,parse' #,depparse,
    ann_str = core_nlp_annotations_str(tokenize, sent_split, postag, parse, dep_parse)
    output = nlp.annotate(text,
        properties={
            'annotators': ann_str,
            'timeout': '50000',
            'outputFormat': 'json'})
    return output

def core_nlp_annotations_str(tokenize=False, sent_split=False, postag=False, parse=False, dep_parse=False):
    ann_flags = []
    # currenty we have to tokenize and sentence split for the JSON processing
    # methods to work properly
    ann_flags.append('tokenize')
    ann_flags.append('ssplit')
    if postag:
        ann_flags.append('pos')
    if dep_parse:
        ann_flags.append('depparse')
    if parse:
        ann_flags.append('parse')
    return ','.join(ann_flags)

def clean_parse(parse):
    parse_str = str(parse)
    parse_str = parse_str.replace('\n', ' ')
    parse_str = re.sub(r' +', ' ', parse_str)
    return parse_str

def main():
    import sys
    if len(sys.argv) >= 2:
        should_tokenize = False
        should_sent_split = False
        should_postag = False
        should_parse = False
        should_dep_parse = False
        if '--tokenize' in sys.argv:
            should_tokenize = sys.argv[sys.argv.index('--tokenize')+1].lower() == 'true'
        if '--sent_split' in sys.argv:
            should_sent_split = sys.argv[sys.argv.index('--sent_split')+1].lower() == 'true'
        if '--postag' in sys.argv:
            should_postag = sys.argv[sys.argv.index('--postag')+1].lower() == 'true'
        if '--parse' in sys.argv:
            should_parse = sys.argv[sys.argv.index('--parse')+1].lower() == 'true'
        if '--dep_parse' in sys.argv:
            should_dep_parse = sys.argv[sys.argv.index('--dep_parse')+1].lower() == 'true'
        corpus = annotate_corpus(read_data(sys.argv[1]),
                               tokenize=should_tokenize,
                               sent_split=should_sent_split,
                               postag=should_postag,
                               parse=should_parse,
                               dep_parse=should_dep_parse)
        print(json.dumps(corpus, indent=2))

if __name__ == "__main__": main()
