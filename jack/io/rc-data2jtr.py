"""

Hermann, Karl Moritz, et al.
"Teaching machines to read and comprehend."
Advances in Neural Information Processing Systems. 2015.

Original paper: https://arxiv.org/abs/1506.03340
Data:   https://github.com/deepmind/rc-data
        http://cs.nyu.edu/~kcho/DMQA/       (direct download)
JTR download script: no download script, check README.md

Metadata:

Number of questions:

        CNN     DailyMail

train   380298  879450
dev     3924    64835
test    3198    53182

"""

import json
import os
import argparse


def create_jtr_snippet(directory, dataset, split, resolve_entities=False, num_instances=5):
    """
    Creates a jack format snippet from rc-data data.

    Args:
        directory: root directory of rc-data
        dataset: which dataset, 'cnn' of 'dailymail'
        split: 'train', 'dev' or 'test'
        resolve_entities: whether to de-anonymise entities. Default: False
        num_instances: number of (first) instances

    Returns:
    """
    return convert_rcdata(directory, dataset, split, resolve_entities, num_instances)


def convert_rcdata(directory, dataset, split, resolve_entities=False, first_n=None):
    """
    Convert subset of rc-data (definet by a combination of dataset and mode) to jtk format
    Args:
        directory: root directory of rc-data
        dataset: which dataset, 'cnn' of 'dailymail'
        split: 'train', 'dev' or 'test'
        resolve_entities: whether to de-anonymise entities. Default: False
        first_n: export a snippet containing the first n instances of the dataset
    Returns:
        jack json

    """
    split_mapping = {'train': 'training', 'test': 'test', 'dev': 'validation'}
    assert split in split_mapping.keys()
    assert dataset in {'cnn', 'dailymail'}

    if directory[-1] != '/':
        directory += "/"
    directory += "{0}/questions/{1}/".format(dataset, split_mapping[split])
    filenames = [file for file in os.listdir(directory) if file.endswith('question')]

    data = {}
    i = 0
    for fname in filenames:
        with open(directory + fname, 'r') as f:
            url = f.readline().strip()
            f.readline()
            text = f.readline().strip()
            f.readline()
            cloze_q = f.readline().strip()
            f.readline()
            answer = f.readline().strip()
            f.readline()
            if resolve_entities:
                entity_dict = {}
                for line in f:
                    split = line.rstrip().split(sep=':')
                    if len(split) != 2:
                        print('Bad split')
                    entity_dict[split[0]] = split[1]
                for key, value in entity_dict.items():
                    text = text.replace(key, value)
                    cloze_q = cloze_q.replace(key, value)
                    answer = answer.replace(key, value)

            if url in data:
                data[url]['rest'].append((fname, cloze_q, answer))
            else:
                data[url] = {
                    'text': text,
                    'rest': [(fname, cloze_q, answer)]
                }
        i += 1
        if first_n and i == first_n:
            break

    instances = []
    counter = 0
    for k, v in data.items():
        url = k
        text = v['text']
        questions = []
        for elem in v['rest']:
            fname, cloze_q, answer = elem
            questions.append({
                'question': cloze_q,
                'answers': [{
                    'text': answer
                }],
                'id': fname
            })
            counter += 1

        instance = {
            # 'id': fname,
            'support': {
                "id": url,
                "text": text
            },
            'questions': questions
        }
        instances.append(instance)

    print(' ...loaded {0} questions.'.format(counter))
    return {
        "meta": "{0}_{1}".format(dataset, split),
        "instances": instances
    }


def main():
    """
    Main call function

    Usage:
        from other code:
            call convert_rcdata(filename)
        from command line:
            call with --help for help

    Returns: nothing
    """
    parser = argparse.ArgumentParser(description='rc-data datasets to jack format converter.')
    parser.add_argument('indir',
                        help="path to the rc-data root directory (e.g. data/rc-data/)")
    parser.add_argument('outfile',
                        help="path to the jack format -generated output file (e.g. data/rc-data/cnn_train.jack.json)")
    parser.add_argument('dataset', choices=['cnn', 'dailymail'],
                        help="which dataset to access: cnn or dailymail")
    parser.add_argument('split', choices=['train', 'dev', 'test'],
                        help="which split of the dataset to io: train, dev or test")
    parser.add_argument('-s', '--snippet', action="store_true",
                        help="Export a snippet (first 5 instances) instead of the full file")
    args = parser.parse_args()

    if args.snippet:
        corpus = create_jtr_snippet(args.indir, args.dataset, args.split, num_instances=5)
    else:
        corpus = convert_rcdata(args.indir, args.dataset, args.split)

    with open(args.outfile, 'w') as outfile:
        json.dump(corpus, outfile, indent=2)


if __name__ == "__main__":
    main()
