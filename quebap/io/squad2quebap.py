import json


def convert_squad(file_path):
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
                    'support': [parse_support(paragraph)],
                    'questions': [parse_question(qa_dict) for qa_dict in paragraph['qas']]
                }
                question_sets.append(qa_set)
    corpus_dict = {
        'meta': {
            'source': filename
        },
        'instances': question_sets
    }
    return corpus_dict


def parse_support(para_dict):
    return {
            'text': para_dict['context']
    }


def parse_question(qa_dict):
    answers = [parse_answer(answer_dict) for answer_dict in qa_dict['answers']]
    return {
        'question': {
            'text': qa_dict['question'],
            'id': qa_dict['id']
        },
        'answers': answers
    }


def parse_answer(answer_dict):
    answer_text = answer_dict['text']
    answer_start = answer_dict['answer_start']
    answer_end = answer_start + len(answer_text)
    return {
        'text': answer_text,
        'span': [answer_start, answer_end]
    }


# Usage:
# From other code, call convert_squad(filename)
# From command line, a single argument converts and writes to stdout
# From command line, two arguments converts arg1 and writes to arg2
def main():
    import sys
    corpus = convert_squad(sys.argv[1])
    if len(sys.argv) == 2:
        print(json.dumps(corpus, indent=2))
    if len(sys.argv) == 3:
        with open(sys.argv[2], 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

if __name__ == "__main__":
    main()
