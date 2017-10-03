import argparse
import csv
import json
from collections import Counter


def convert_newsqa(file_path):
    """
    Converts NewsQA dataset to jack format.

    Args:
        file_path: path to the NewsQA CSV file (data/NewsQA/)

    Returns: dictionary in jack format
    """
    # meta info
    if '/' in file_path:
        filename = file_path[file_path.rfind('/') + 1:]  # Maybe support a system-specific delimiter
    else:
        filename = file_path

    # data
    question_sets = []
    with open(file_path) as data_file:
        reader = csv.reader(data_file)
        reader.__next__()
        for row in reader:
            [story_id, question, answer_char_ranges, is_answer_absent, is_question_bad, validated_answers,
             story_text] = row

            if validated_answers:
                answers = json.loads(validated_answers)
                spans = [k for k, v in answers.items() if ":" in k]
            else:
                answers = Counter()
                for rs in answer_char_ranges.split("|"):
                    for r in set(rs.split(",")):
                        if ":" in r:
                            answers[r] += 1
                spans = [k for k, v in answers.items() if ":" in k and v >= 2]

            if spans:
                qa_set = {
                    "support": [story_text],
                    "questions": [{
                        'question': {
                            'text': question,
                            'id': story_id + "_" + question.replace(" ", "_")
                        },
                        'answers': [{"span": [int(span.split(":")[0]), int(span.split(":")[1])],
                                     "text": story_text[int(span.split(":")[0]):int(span.split(":")[1])]
                                     } for span in spans]
                    }]
                }
                question_sets.append(qa_set)

    corpus_dict = {
        'meta': {
            'source': filename
        },
        'instances': question_sets
    }

    return corpus_dict


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
    parser = argparse.ArgumentParser(description='NewsQA dataset to jack format converter.')
    parser.add_argument('infile',
                        help="path to the input file, original NewsQA file (e.g. data/NewsQA/newsqa/maluuba/newsqa/split_data/train.csv)")
    parser.add_argument('outfile',
                        help="path to the jack format -generated output file (e.g. data/NewsQA/train.jack.json)")

    args = parser.parse_args()
    corpus = convert_newsqa(args.infile)

    with open(args.outfile, 'w') as outfile:
        json.dump(corpus, outfile, indent=2)


if __name__ == "__main__":
    main()
