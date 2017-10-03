"""

Hill, Felix, et al.
"The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations."
arXiv preprint arXiv:1511.02301 (2015).


Original paper: https://arxiv.org/abs/1511.02301
Data: https://research.fb.com/projects/babi/
JTR download script: data/CBT/download.sh

"""

import json
import argparse


def __load_cbt_file(path=None, part='train', mode='NE'):
    """
    NOT USED RIGHT NOW

    Path should be given and function will load raw data.
    If it is no given however, there's three parts:
    'train', 'valid' and 'test' as well as 5 modes.
    The modes are:
            - 'CN' (predicting common nouns)
            - 'NE' (predicting named entities)
            - 'P'  (predicting prepositions)
            - 'V'  (predicting verbs.)
            - 'all'(all of the above four categories)
    When calling this function both the dataset part and the mode have to
    be specified.
    'all' is not suitable for QA format, it seems to be just raw text.

    Args:
        path:
        part:
        mode:

    Returns:

    """
    if path is None:
        if mode == 'all':
            path += 'cbt_' + part + '.txt'
        else:
            path += 'cbtest_' + mode + '_' + part
            if part == 'valid':
                path += '_2000ex.txt'
            elif part == 'test':
                path += '_2500ex.txt'
    with open(path, 'r') as f:
        data = f.read()
    return data


def __split_cbt(raw_data, first_n=None):
    """ splits raw cbt data into parts corresponding to each instance """
    story_instances = []
    instance = []
    for l in raw_data.split('\n')[:-1]:
        if l == '':  # reset instance every time an empty line is encountered
            story_instances.append(instance)
            instance = []
            continue
        instance.append(l)
    if first_n:
        return story_instances[:first_n]
    return story_instances


def __parse_cbt_example(instance):
    support = question = answer = candidates_string = []
    for line in instance:
        line_number, line_content = line.split(" ", 1)
        if int(line_number) < 21:    # line contains sentence
            support.append(line_content)
        else:
            question, answer, candidates_string = line_content.split('\t', 2)
    candidates_list = candidates_string.strip('\t').split('|')
    qdict = {
        'question': question,
        'candidates': [{'text': cand} for cand in candidates_list],
        'answers': [{'text': answer}]
    }
    questions = [qdict]
    qset_dict = {
        'suport': [{'text': supp} for supp in support],
        'questions': questions
    }

    return qset_dict


def create_jtr_snippet(path, n_instances=5):
    """
    Creates a jack format snippet.

    Args:
        path: path to the file
        n_instances: number of instances

    Returns: jack json

    """
    return convert_cbt(path, n_instances)


def convert_cbt(path, n_instances=None):
    """
    Convert the Children's Book Test file into jack format.

    Args:
        path: the file which should be converted
        n_instances: how many instances to filter out
    Returns: dictionary in jack format

    """
    # raw_data = __load_cbt_file(path)
    with open(path, 'r') as f:
        raw_data = f.read()

    instances = __split_cbt(raw_data, n_instances)

    corpus = []
    for inst in instances:
        corpus.append(__parse_cbt_example(inst))

    return {
        'meta': 'Children\'s Book Test',
        'globals': {'candidates': []},
        'instances': corpus
    }


def main():
    """
        Main call function

    Usage:
        from other code:
            call convert_cbt(filename)
        from command line:
            call with --help for help

    Returns: nothing
    """

    parser = argparse.ArgumentParser(description='The Children’s Book Test (CBT) dataset to jack format converter.')
    parser.add_argument('infile',
                        help="path to the file to be converted (e.g. data/CBT/CBTest/data/cbtest_CN_train.txt)")
    parser.add_argument('outfile',
                        help="path to the jack format -generated output file (e.g. data/CBT/train.jack.json)")
    parser.add_argument('-s', '--snippet', action="store_true",
                        help="Export a snippet (first 5 instances) instead of the full file")
    args = parser.parse_args()

    if args.snippet:
        corpus = convert_cbt(args.infile, n_instances=5)
    else:
        corpus = convert_cbt(args.infile)
    with open(args.outfile, 'w') as outfile:
        json.dump(corpus, outfile, indent=2)

if __name__ == "__main__":
    main()
