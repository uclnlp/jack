import json
import argparse
import re

err_count = 0
count = 0


def create_snippet(file_path):
    """
    Creates a snippet of the original SciQ data.

    Args:
        file_path: path to the original file

    Returns: string containing file contents
    """
    with open(file_path) as data_file:
        data = json.load(data_file)['data']
        out = {
            'data': [{
                'title': data[0]['title'],
                'paragraphs': [{
                    'context': data[0]['paragraphs'][0]['context'],
                    'qas': data[0]['paragraphs'][0]['qas'][0:3]
                }]
            }]
        }
        return json.dumps(out, indent=2)


def create_jtr_snippet(jtr_dict, num_instances=1):
    """
    Creates a jtr format snippet from SciQ data.

    Args:
        jtr_dict: jtr dictionary
        num_instances: number of (first) instances

    Returns: dictionary in jtr format
    """
    out = dict()
    out['meta'] = jtr_dict['meta']
    out['instances'] = jtr_dict['instances'][0:num_instances]
    return out


def convert_sciq(file_path, mult_answer=0):
    """
    Converts SciQ dataset to jtr format.

    Args:
        file_path: path to the SciQ json file (train-v4.json and dev-v4.json in data/SciQ/)

    Returns: dictionary in jtr format
    """
    global count
    global err_count
    # meta info
    if '/' in file_path:
        filename = file_path[file_path.rfind('/')+1:]   # Maybe support a system-specific delimiter
    else:
        filename = file_path
    # data
    question_sets = []
    with open(file_path) as data_file:
        data = json.load(data_file)
        for question in data:
            count += 1
            q = __parse_question(question, mult_answer)
            if q is None:
                err_count += 1
                continue
            qa_set = {
                'support': [__parse_support(question)],
                'questions': [q]
            }
            question_sets.append(qa_set)
    corpus_dict = {
        'meta': {
            'source': filename
        },
        'instances': question_sets
    }
    return corpus_dict


def __parse_support(para_dict):
    return {
            'text': para_dict['support']
    }


def __parse_question(qa_dict, mult_answer):
    answers = __parse_answer(qa_dict['correct_answer'], qa_dict['support'])
    if answers is None:
        return None

    if not mult_answer:
        answers = [answers[0]]

    return {
        'question': {
            'text': qa_dict['question']
        },
        'answers': answers
    }


def __parse_answer(answer, support):
    answers = [{'text': answer, 'span': [m.start(0), m.end(0)]} for m in re.finditer(re.escape(answer), support, re.IGNORECASE)]

    if len(answers) == 0:
        return None

    return answers


def main():
    """
    Main call function

    Usage:
        from other code:
            call convert_sciq(filename)
        from command line:
            call with --help for help

    Returns: nothing
    """
    parser = argparse.ArgumentParser(description='SciQ dataset to jtr format converter.')
    parser.add_argument('infile',
                        help="path to the input file, original SciQ file (e.g. data/SciQ/train.v4.json)")
    parser.add_argument('outfile',
                        help="path to the jtr format -generated output file (e.g. data/SciQ/train.jtr.json)")
    parser.add_argument('-s', '--snippet', action="store_true",
                        help="Export a snippet (first 5 instances) instead of the full file")
    parser.add_argument('--mult_answer', type=int, default=0, required=False,
                        help="when 0 will choose the first occurency of the answer in the " +
                             "support document as correct answer span, when 1 if multiple occurency " +
                             "are found multiple candidate span are craeted")

    args = parser.parse_args()
    corpus = convert_sciq(args.infile, args.mult_answer)
    if args.snippet:
        corpus = create_jtr_snippet(corpus, num_instances=5)

    with open(args.outfile, 'w') as outfile:
        json.dump(corpus, outfile, indent=2)


if __name__ == "__main__":
    main()
    print('{0}/{1} questions imported correctly'.format((count-err_count), count))
