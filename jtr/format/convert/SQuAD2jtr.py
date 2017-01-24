import json


def create_snippet(file_path):
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


def create_jtr_snippet(jtr_dict):
    out = {
        'meta': jtr_dict['meta'],
        'instances': [{
            'support': jtr_dict['instances'][0]['support'],
            'questions':jtr_dict['instances'][0]['questions'][0:3]
        }]
    }
    return json.dumps(out, indent=2)


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
        'span': [answer_start, answer_end]
    }


def main():
    """
    Usage:
    From other code, call convert_squad(filename)
    NOT APPLICABLE - From command line, a single argument converts and writes to stdout
    From command line, two arguments converts arg1 and writes to arg2

    Returns: nothing

    """
    import sys
    # if len(sys.argv) == 2:
    #     print(json.dumps(corpus, indent=2))
    if len(sys.argv) == 3:
        corpus = convert_squad(sys.argv[1])
        with open(sys.argv[2], 'w') as outfile:
            json.dump(corpus, outfile, indent=2)
    else:
        print("Usage: python3 SQuAD2jtr.py path/to/SQuAD save/to/SQuAD.jtr.json")


if __name__ == "__main__":
    main()
