"""
This files allows creating a quebap datafile for the SNLI corpus,
whereby each instance receives support under the form of 'related' instances 
"""


import json

__candidate_labels = ['entailment','neutral','contradiction']    
__candidates = [{'text':cl} for cl in __candidate_labels]


def convert_snli(snli_file_jsonl):
    """ convert SNLI files into quebap format. 
    Data source: http://nlp.stanford.edu/projects/snli/snli_1.0.zip
    Files to be converted: snli_1.0_dev.jsonl, snli_1.0_train.jsonl, snli_1.0_test.jsonl
    (the *.txt files contain the same data in a different format)

    Format:
        - support = the premise = 'sentence1' in original SNLI data (as id, use 'captionID', the id of sentence1) 
        - question = the hypothesis = 'sentence2' in original SNLI data
    Notes:
        - instances with gold labels '-' are removed from the corpus
    """
    assert 'snli_1.0' in snli_file_jsonl and snli_file_jsonl.endswith('.jsonl'), "input should be the snli_1.0_X.jsonl files (X=test/train/dev)"

    with open(snli_file_jsonl,'r') as f:
        data = [__convert_snli_instance(json.loads(line.strip())) for line in f.readlines()]

        return {'meta': 'SNLI',
                'globals': {'candidates': __candidates},
                'instances': [d for d in data if d]  # filter out invalid ones
                }


def __convert_snli_instance(instance):
    try:
        if not instance['gold_label'] in __candidate_labels:
            raise IOError('invalid gold label')
        queb = {}
        queb['id'] = instance['pairID']
        queb['support'] = [
            {'id': instance['captionID'], 'text': instance['sentence1']}]
        queb['questions'] = [
            {'question': instance['sentence2'],
             'answers': [
                 # {'index': __candidate_labels.index(instance['gold_label'])},
                 {'text': __candidate_labels[__candidate_labels.index(instance['gold_label'])]}]}]

        return queb

    except IOError:
        return None


def main():
    import sys
    if len(sys.argv) == 2:
        corpus = convert_snli(sys.argv[1])
        print(json.dumps(corpus, indent=2))
    else:
        for corpus_name in ["train", "dev", "test"]:
            corpus = convert_snli("./quebap/data/SNLI/snli_1.0/snli_1.0_%s.jsonl" % corpus_name)
            with open("./quebap/data/SNLI/snli_1.0/snli_1.0_%s_quebap.jsonl" % corpus_name, 'w') as outfile:
                json.dump(corpus, outfile, indent=2)

        # create snippet
        corpus = convert_snli("./quebap/data/SNLI/snli_1.0/snli_1.0_train.jsonl")
        corpus['instances'] = corpus['instances'][:10]
        with open("./quebap/data/SNLI/snli_1.0/snli_1.0_debug_quebap_v1.jsonl", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)
        with open("./quebap/data/SNLI/snippet_quebapformat_v1.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

if __name__ == "__main__":
    main()
