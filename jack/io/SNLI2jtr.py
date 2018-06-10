"""
This files allows creating a jtr datafile for the SNLI corpus,
whereby each instance receives support under the form of 'related' instances
"""


import json

__candidate_labels = ['entailment', 'neutral', 'contradiction']
__candidates = [{'text': cl} for cl in __candidate_labels]


def convert_snli(snli_file_jsonl):
    """ io SNLI files into jack format.
    Data source: http://nlp.stanford.edu/projects/snli/snli_1.0.zip
    Files to be converted: snli_1.0_dev.jsonl, snli_1.0_train.jsonl, snli_1.0_test.jsonl
    (the *.txt files contain the same data in a different format)

    Format:
        - support = the premise = 'sentence1' in original SNLI data (as id, use 'captionID', the id of sentence1)
        - question = the hypothesis = 'sentence2' in original SNLI data
    Notes:
        - instances with gold labels '-' are removed from the corpus
    """
    with open(snli_file_jsonl, 'r') as f:
        data = [__convert_snli_instance(json.loads(line.strip())) for line in f.readlines()]

        return {'meta': 'SNLI',
                'globals': {'candidates': __candidates},
                'instances': [d for d in data if d]  # filter out invalid ones
                }


def __convert_snli_instance(instance):
    queb = {}
    if instance['gold_label'] in __candidate_labels:
        queb['id'] = instance['pairID']
        queb['support'] = [
            {'id': instance.get('captionID'), 'text': instance['sentence1']}]
        queb['questions'] = [
            {'question': instance['sentence2'],
             'answers': [{'text': instance['gold_label']}]}]
    return queb


def main():
    import sys
    if len(sys.argv) == 2:
        corpus = convert_snli(sys.argv[1])
    else:
        for corpus_name in ["dev", "train", "test"]:
            corpus = convert_snli("./data/SNLI/snli_1.0/snli_1.0_%s.jsonl" % corpus_name)
            with open("./data/SNLI/snli_1.0/snli_1.0_%s_jtr_v1.json" % corpus_name, 'w') as outfile:
                print("Create file snli_1.0_%s_jtr.json" % corpus_name)
                json.dump(corpus, outfile, indent=2)

        # create train set test data
        corpus = convert_snli("./data/SNLI/snli_1.0/snli_1.0_train.jsonl")
        corpus['instances'] = corpus['instances'][:2000]
        with open("./tests/test_data/SNLI/2000_samples_train_jtr_v1.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        corpus['instances'] = corpus['instances'][:100]
        with open("./tests/test_data/SNLI/overfit.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        # create snippets and overfit test data
        corpus['instances'] = corpus['instances'][:10]
        with open("./data/SNLI/snli_1.0/snli_1.0_debug_jtr_v1.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)
        with open("./data/SNLI/snippet.jtr_v1.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        # create dev set test data
        corpus = convert_snli("./data/SNLI/snli_1.0/snli_1.0_dev.jsonl")
        corpus['instances'] = corpus['instances'][:1000]
        with open("./tests/test_data/SNLI/1000_samples_dev_jtr_v1.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        # create dev set test data
        corpus = convert_snli("./data/SNLI/snli_1.0/snli_1.0_test.jsonl")
        corpus['instances'] = corpus['instances'][:2000]
        with open("./tests/test_data/SNLI/2000_samples_test_jtr_v1.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

if __name__ == "__main__":
    main()
