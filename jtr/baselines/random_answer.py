# -*- coding: utf-8 -*-

import json
import random
import sys


def get_answers(data):
    deep_answers = [get_answers_to_qset(qset) for qset in data]
    flat_answers = [item for sublist in deep_answers for item in sublist]
    return flat_answers


def get_answers_to_qset(qset):
    return [[random.choice(q['candidates'])] for q in qset['questions']]


# Should be a call to an external read code for jtr format
def read_data(data_filename):
    with open(data_filename) as data_file:
        data = json.load(data_file)
        return data


def main():
    random.seed(1)
    if len(sys.argv) == 2:
        data = read_data(sys.argv[1])
        answers = get_answers(data)
        print(json.dumps(answers, indent=2))

if __name__ == "__main__":
    main()
