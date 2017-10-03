"""

Richardson, Matthew, Christopher JC Burges, and Erin Renshaw.
"MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text."
EMNLP. Vol. 3. 2013.

Original paper: http://research.microsoft.com/en-us/um/redmond/projects/mctest/MCTest_EMNLP2013.pdf
Data:   research.microsoft.com/en-us/um/redmond/projects/mctest/
JTR download script: data/MCTest/download.sh

"""

import json
import argparse

labels = ["A", "B", "C", "D"]


def __clean_mctest_text(text):
    return text.replace('\\newline', '  ')


def create_jtr_snippet(tsv_file, ans_file, n_instances=5):
    return convert_mctest(tsv_file, ans_file, n_instances)


def convert_mctest(tsv_file, ans_file, first_n=None):
    with open(tsv_file) as tsv_data:
        tsv_lines = tsv_data.readlines()
    with open(ans_file) as ans_data:
        ans_lines = ans_data.readlines()
        instances = []
    for tsv, ans in zip(tsv_lines, ans_lines):
        instances.append(__parse_mctest_instance(tsv, ans))
    if first_n:
        instances = instances[:first_n]
    return {
        "meta": "MCTest",
        "instances": instances
    }



def __parse_mctest_instance(tsv_chunk, ans_chunk):
    tsv_tabs = tsv_chunk.strip().split('\t')
    ans_tabs = ans_chunk.strip().split('\t')
    # id = tsv_tabs[0]
    # ann = tsv_tabs[1]
    passage = tsv_tabs[2]
    # the dictionary for populating a set of passage/questions/answers
    qset_dict = {
        'support': [{'text': __clean_mctest_text(passage)}],
        'questions': __parse_mctest_questions(tsv_tabs[3:], ans_tabs)
    }
    return qset_dict


def __parse_mctest_questions(question_list, ans_tabs):
    # print(ans_tabs)
    questions = []
    for i in range(0, len(question_list), 5):
        # qdict = {}
        # parse answers
        candidates = []
        correct_answer = ans_tabs[int(i / 5)]
        for j in range(1, 5):
            label = labels[j-1]
            answer = {
                'label': label,
                'text': question_list[i + j]
            }
            candidates.append(answer)
        correct_index = labels.index(correct_answer)
        answer = {
            'index': correct_index,
            'text': question_list[i + correct_index + 1]
        }
        # parse question
        qcols = question_list[i].split(':')
        qdict = {
            'question': qcols[1],
            'candidates': candidates,
            'answers': [answer]
        }
        questions.append(qdict)
    return questions


def main():
    parser = argparse.ArgumentParser(description='Machine Comprehension Test (MCTest) dataset to jack format converter.')
    parser.add_argument('in_tsv',
                        help="path to the MCTest tsv file (e.g. data/MCTest/MCTest/mc160.train.tsv)")
    parser.add_argument('in_ans',
                        help="path to the MCTest ans file (e.g. data/MCTest/MCTest/mc160.train.ans)")
    parser.add_argument('outfile',
                        help="path to the jack format -generated output file (e.g. data/MCTest/train.160.jack.json)")
    parser.add_argument('-s', '--snippet', action="store_true",
                        help="Export a snippet (first 5 instances) instead of the full file")
    args = parser.parse_args()


    if args.snippet:
        corpus = convert_mctest(args.in_tsv, args.in_ans, 5)
    else:
        corpus = convert_mctest(args.in_tsv, args.in_ans)

    with open(args.outfile, 'w') as outfile:
        json.dump(corpus, outfile, indent=2)

if __name__ == "__main__":
    main()
