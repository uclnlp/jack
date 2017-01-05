import json


def load_cbt_file(path=None, part='train', mode='NE'):
    """ Path should be given and function will load raw data.
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
    """
    if path is None:
        # path='/Users/Johannes/PhD/kebab/pre/CBTest/data/'
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


def split_cbt(raw_data):
    """
    splits raw cbt data into parts corresponding to each instance
    :param raw_data:
    :return:
    """
    story_instances = []
    instance = []
    for l in raw_data.split('\n')[:-1]:
        if l == '':  # reset instance every time an empty line is encountered
            story_instances.append(instance)
            instance = []
            continue
        instance.append(l)
    return story_instances[:2]


def parse_cbt_example(instance):
    """
    returns dictionary with support, question, correct answer and candidates.
    :param instance:
    :return:
    """
    # D = {}
    support = []
    for line in instance:
        line_number, line_content = line.split(" ", 1)
        if int(line_number) < 21:    # line contains sentence
            support.append(line_content)
        else:
            question, answer, candidates_string = line_content.split('\t', 2)
    candidates_list = candidates_string.strip('\t').split('|')
    qdict = {
        'question': question,
        'candidates': [
            {
                'text' : cand
            } for cand in candidates_list
        ],
        'answers': [{'text': answer}]
    }
    questions = [qdict]
    qset_dict = {
                    'suport': [{'text': supp} for supp in support],
                    'questions': questions
                }

    return qset_dict


def convert_cbtest(path):
    """
    convert Children Book Test file into quebap format. Path should indicate
    the file which should be converted, e.g.
    :param path: '.../data/cbtest_CN_train.txt'
    :return:
    """
    raw_data = load_cbt_file(path)
    instances = split_cbt(raw_data)
    corpus = []
    for inst in instances:
        corpus.append(parse_cbt_example(inst))

    return {'meta': 'Children Book Test',
           'globals': {'candidates': []},
           'instances': corpus
           }


def main():
    # raw_data = load_cbt_file(path=None, part='valid', mode='NE')
    # instances = split_cbt(raw_data)
    #_ = parse_cbt_example(instances[0])
    """
    Usage: provide path to CBT data file as single argument, e.g. '.../data/cbtest_CN_train.txt'
    """

    import sys
    if len(sys.argv) == 2:
        corpus = convert_cbtest(path=sys.argv[1])
        print(json.dumps(corpus, indent=2))

if __name__ == "__main__":
    main()
