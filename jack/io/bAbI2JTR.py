"""

Weston et al. 2015
"Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
arXiv preprint https://arxiv.org/abs/1502.05698.

Data: https://research.fb.com/projects/babi/
JTR download script: data/bAbI/download.sh

"""

import json
import argparse

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def load_babi_task(filename):
    """
    loads the babi data from its original format.
    """
    tasks = []
    task = None
    for i, line in enumerate(open(filename, 'r')):
        ID = int(line[0:line.find(' ')])

        # new data example
        if ID == 1:
            current_story = []
            counter = 0
            id_map = {}

        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]

        if '\t' not in line:
            # if line is not a question
            current_story.append(line)
            id_map[ID] = counter
            counter += 1
        else:
            # if the line is a question
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            question = line[:idx]
            answer = tmp[1].strip()
            # copy by value.
            this_task = {"Story": [x for x in current_story], "Question": question, "Answer": answer}
            tasks.append(this_task)

    return tasks



def single_babi_example_in_JTR_format(instance):
    candidates_list = []
    qdict = {
        'question': instance['Question'],
        'candidates': [{'text': cand} for cand in candidates_list],
        'answers': [{'text': instance['Answer']}]
    }
    questions = [qdict]
    qset_dict = {
        'support': [{'text': supp} for supp in instance["Story"]],
        'questions': questions
    }

    return qset_dict



def convert_babi(path, n_instances=None):
    """
    Convert the babi data into jack format.

    Args:
        path: the file which should be converted
        n_instances: how many instances to filter out
    Returns: dictionary in jack format

    """
    # load data, select only first few if required.
    babi_data = load_babi_task(path)
    if n_instances != None:
        babi_data = babi_data[:n_instances]


    corpus = []
    for instance in babi_data:
        corpus.append(single_babi_example_in_JTR_format(instance))

    return {
        'meta': 'bAbI',
        'globals': {'candidates': []},
        'instances': corpus
    }


def main():
    """
        Main call function

    Usage:
        from command line:
            call with --help for help

    Returns: nothing
    """

    parser = argparse.ArgumentParser(description='The bAbI dataset to jack format converter.')
    parser.add_argument('infile',
                        help="path to the file to be converted (e.g. data/bAbI/tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt)")
    parser.add_argument('outfile',
                        help="path to the jack format -generated output file (e.g. data/bAbI/jtr_format/train.jack.json)")
    parser.add_argument('-s', '--snippet', action="store_true",
                        help="Export a snippet (first 5 instances) instead of the full file")
    args = parser.parse_args()

    if args.snippet:
        jtr_corpus = convert_babi(args.infile, n_instances=4)
    else:
        jtr_corpus = convert_babi(args.infile)
    with open(args.outfile, 'w') as outfile:
        json.dump(jtr_corpus, outfile, indent=2)

if __name__ == "__main__":
    main()
