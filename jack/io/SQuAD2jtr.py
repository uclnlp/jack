"""

Rajpurkar, Pranav, et al.
"Squad: 100,000+ questions for machine comprehension of text."
arXiv preprint arXiv:1606.05250 (2016).

Original paper: https://arxiv.org/abs/1606.05250
Data: https://rajpurkar.github.io/SQuAD-explorer/
JTR download script: data/SQuAD/download.sh

"""

import argparse
import json


def create_snippet(file_path):
    """
    Creates a snippet of the original SQuAD data.

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
    Creates a jack format snippet from SQuAD data.

    Args:
        jtr_dict: jack dictionary
        num_instances: number of (first) instances

    Returns: dictionary in jack format
    """
    out = dict()
    out['meta'] = jtr_dict['meta']
    out['instances'] = jtr_dict['instances'][0:num_instances]
    return out


def convert_squad(file_path):
    """
    Converts SQuAD dataset to jack format.

    Args:
        file_path: path to the SQuAD json file (train-v1.1.json and dev-v1.1.json in data/SQuAD/)

    Returns: dictionary in jack format
    """
    # meta info
    if '/' in file_path:
        filename = file_path[file_path.rfind('/')+1:]   # Maybe support a system-specific delimiter
    else:
        filename = file_path
    # data
    question_sets = []
    with open(file_path) as data_file:
        data = json.load(data_file)['data']
        for article in data:
            for paragraph in article['paragraphs']:
                qa_set = {
                    'support': [__parse_support(paragraph)],
                    'questions': [__parse_question(qa_dict) for qa_dict in paragraph['qas']]
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
            'text': para_dict['context']
    }


def __parse_question(qa_dict):
    answers = [__parse_answer(answer_dict) for answer_dict in qa_dict['answers']]
    return {
        'question': {
            'text': qa_dict['question'],
            'id': qa_dict['id']
        },
        'answers': answers
    }


def __parse_answer(answer_dict):
    answer_text = answer_dict['text']
    answer_start = answer_dict['answer_start']
    answer_end = answer_start + len(answer_text)
    return {
        'text': answer_text,
        'span': (answer_start, answer_end),
        'doc_idx': 0,  # in SQuAD there is always only a single document
    }


def main():
    """
    Main call function

    Usage:
        from other code:
            call convert_squad(filename)
        from command line:
            call with --help for help

    Returns: nothing
    """
    parser = argparse.ArgumentParser(description='SQuAD dataset to jack format converter.')
    parser.add_argument('infile',
                        help="path to the input file, original SQuAD file (e.g. data/SQuAD/train-v1.1.json)")
    parser.add_argument('outfile',
                        help="path to the jack format -generated output file (e.g. data/SQuAD/train.jack.json)")
    parser.add_argument('-s', '--snippet', action="store_true",
                        help="Export a snippet (first 5 instances) instead of the full file")

    args = parser.parse_args()
    corpus = convert_squad(args.infile)
    if args.snippet:
        corpus = create_jtr_snippet(corpus, num_instances=5)

    with open(args.outfile, 'w') as outfile:
        json.dump(corpus, outfile, indent=2)


if __name__ == "__main__":
    main()
