"""
This files allows creating a jtr datafile for the SNLI corpus,
whereby
- questions consist of premise + delim. + hypothesis
- TODO: support facts can optinally be loaded from Wordnet / SNLItraining / PPDB
"""


import json

__candidate_labels = ['entailment','neutral','contradiction']
__candidates = [{'text':cl} for cl in __candidate_labels]

CONJ = '|||||'


def convert_snli(snli_file_jsonl, support=False):
    """ io SNLI files into jack format.
    Data source: http://nlp.stanford.edu/projects/snli/snli_1.0.zip
    Files to be converted: snli_1.0_dev.jsonl, snli_1.0_train.jsonl, snli_1.0_test.jsonl
    (the *.txt files contain the same data in a different format)

    Format:
        - question = the premise + CONJ + the hypothesis  = 'sentence1' + CONJ + 'sentence2'
        - support instance = part1 + CONJ + part2 + CONJ + label
    Notes:
        - instances with gold labels '-' are removed from the corpus

    Args:
        snli_file_json: input file
        support: False (no support), TODO: 'WordNet', 'SNLItrain', 'PPDB'
    """
    assert 'snli_1.0' in snli_file_jsonl and snli_file_jsonl.endswith('.jsonl'), "input should be the snli_1.0_X.jsonl files (X=test/train/dev)"

    with open(snli_file_jsonl,'r') as f:
        data = [__convert_snli_instance(json.loads(line.strip())) for line in f.readlines()]
        instances = __add_support([d for d in data if d], support)

        return {'meta': 'SNLI',
                'globals': {'candidates': __candidates},
                'instances': instances  # filter out invalid ones
                }


def __convert_snli_instance(instance):
    try:
        if not instance['gold_label'] in __candidate_labels:
            raise IOError('invalid gold label')
        queb = {}
        queb['id'] = instance['pairID']
        queb['support'] = []
        queb['questions'] = [
            {'question': instance['sentence1'] + CONJ + instance['sentence2'],
             'answers': [
                 {'text': __candidate_labels[__candidate_labels.index(instance['gold_label'])]}]}]

        return queb

    except IOError:
        return None


def __add_support(instances, support):
    """
    Args:
        instances: list of jack instances (with or without support)
        support: False (no support), TODO: 'WordNet', 'SNLItrain', 'PPDB'
    """

    if support in ['WordNet', 'SNLItrain', 'PPDB']:
        # TODO: add support
        pass

    return instances


def main():
    import sys
    if len(sys.argv) == 2:
        corpus = convert_snli(sys.argv[1])
        print(json.dumps(corpus, indent=2))
    else:
        for corpus_name in ["dev","train","test"]:
            corpus = convert_snli("./jack/data/SNLI/snli_1.0/snli_1.0_%s.jsonl" % corpus_name, support=False)
            with open("./jack/data/SNLI/snli_1.0/snli_1.0_%s_jtr_v2.json" % corpus_name, 'w') as outfile:
                json.dump(corpus, outfile, indent=2)

        # create snippet
        corpus = convert_snli("./jack/data/SNLI/snli_1.0/snli_1.0_train.jsonl", support=False)
        corpus['instances'] = corpus['instances'][:10]
        with open("./jack/data/SNLI/snli_1.0/snli_1.0_debug_jtr_v2.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)
        with open("./jack/data/SNLI/snippet_jtrformat_v2.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

if __name__ == "__main__":
    main()
