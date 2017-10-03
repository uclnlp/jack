"""
This files allows creating a jtr datafile for the SNLI corpus,
whereby each instance receives support under the form of 'related' instances
"""


import json

__candidate_labels = ['entailment', 'neutral', 'contradiction']
__candidates = [{'text': cl} for cl in __candidate_labels]


def convert_snli(multisnli_file):
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
    assert 'multinli_0.9' in multisnli_file and multisnli_file.endswith('.txt'),\
        "input should be the multinli_0.9_X.txt files (X=test/train/dev)"

    with open(multisnli_file, 'r') as f:
        data = [__convert_snli_instance(line.strip().strip("\n").split("\t")) for line in f.readlines()]

        return {'meta': 'MultiSNLI',
                'globals': {'candidates': __candidates},
                'instances': [d for d in data if d]  # filter out invalid ones
                }


def __convert_snli_instance(lspl):
    if len(lspl) != 15:
        return None
    try:
        gold_label, _, _, _, _, sentence1, sentence2, promptID, pairID, genre, _, _, _, _, _ = lspl
        if gold_label == "gold_label":
            return None

        if not gold_label in __candidate_labels:
            raise IOError('invalid gold label')
        queb = {}
        queb['id'] = pairID
        queb['genre'] = genre,
        queb['support'] = [
            {'id': promptID, 'text': sentence1}]
        queb['questions'] = [
            {'question': sentence2,
             'answers': [
                 # {'index': __candidate_labels.index(instance['gold_label'])},
                 {'text': __candidate_labels[__candidate_labels.index(gold_label)]}]}]

        return queb

    except IOError:
        return None


def main():
    import sys
    if len(sys.argv) == 2:
        corpus = convert_snli(sys.argv[1])
    else:
        for corpus_name in ["dev_matched", "train"]:
            corpus = convert_snli("../../data/MultiNLI/multinli_0.9_%s.txt" % corpus_name)
            with open("../../data/MultiNLI/multinli_0.9_%s_jtr.json" % corpus_name, 'w') as outfile:
                print("Create file snli_0.9_%s_jtr.txt" % corpus_name)
                json.dump(corpus, outfile, indent=2)

        # create train set test data
        corpus = convert_snli("../../data/MultiNLI/multinli_0.9_train.txt")
        corpus['instances'] = corpus['instances'][:2000]
        with open("../../tests/test_data/MultiNLI/2000_samples_train_jtr.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        corpus['instances'] = corpus['instances'][:100]
        with open("../../tests/test_data/MultiNLI/overfit.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        # create snippets and overfit test data
        corpus['instances'] = corpus['instances'][:10]
        with open("../../data/MultiNLI/multinli_0.9_debug_jtr.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)
        with open("../../data/MultiNLI/snippet.jack.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        # create dev set test data
        corpus = convert_snli("../../data/MultiNLI/multinli_0.9_dev_matched.txt")
        corpus['instances'] = corpus['instances'][:1000]
        with open("../../tests/test_data/MultiNLI/1000_samples_dev_jtr.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        # create dev set test data
        corpus = convert_snli("../../data/MultiNLI/multinli_0.9_train.txt")
        corpus['instances'] = corpus['instances'][:2000]
        with open("../../tests/test_data/MultiNLI/2000_samples_train_jtr.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)


if __name__ == "__main__":
    main()