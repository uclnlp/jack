import json
import os


def convert_rcdata(directory, dataset, mode, resolve_entities=False):
    """
    Convert subset of rc-data (definet by a combination of dataset and mode) to jtk format
    Args:
        directory: root directory of rc-data
        dataset: which dataset, 'cnn' of 'dailymail'
        mode: 'train', 'dev' or 'test'
        resolve_entities: whether to de-anonymise entities. Default: False

    Returns:
        jtr json

    """
    mode_mapping = {'train': 'training', 'test': 'test', 'dev': 'validation'}
    assert mode in mode_mapping.keys()

    if directory[-1] != '/':
        directory += "/"
    directory += "{0}/questions/{1}/".format(dataset, mode_mapping[mode])
    filenames = [file for file in os.listdir(directory) if file.endswith('question')]

    data = {}
    for fname in filenames:
        with open(directory + fname, 'r') as f:
            url = f.readline().strip()
            f.readline()
            text = f.readline().strip()
            f.readline()
            cloze_q = f.readline().strip()
            f.readline()
            answer = f.readline().strip()
            f.readline()
            if resolve_entities:
                entity_dict = {}
                for line in f:
                    split = line.rstrip().split(sep=':')
                    if len(split) != 2:
                        print('Bad split')
                    entity_dict[split[0]] = split[1]
                for key, value in entity_dict.items():
                    text = text.replace(key, value)
                    cloze_q = cloze_q.replace(key, value)
                    answer = answer.replace(key, value)

            if url in data:
                data[url]['rest'].append((fname, cloze_q, answer))
            else:
                data[url] = {
                    'text': text,
                    'rest': [(fname, cloze_q, answer)]
                }

    instances = []
    counter = 0
    for k, v in data.items():
        url = k
        text = v['text']
        questions = []
        for elem in v['rest']:
            fname, cloze_q, answer = elem
            questions.append({
                'question': cloze_q,
                'answers': [{
                    'text': answer
                }],
                'id': fname
            })
            counter += 1

        instance = {
            # 'id': fname,
            'support': {
                "id": url,
                "text": text
            },
            'questions': questions
        }
        instances.append(instance)

    print(' ...loaded {0} questions.'.format(counter))
    return {
        "meta": "{0}_{1}".format(dataset, mode),
        "instances": instances
    }


def main():
    import sys

    if len(sys.argv) == 5:
        dataset = sys.argv[1]
        mode = sys.argv[2]
        directory = sys.argv[3]
        outfname = sys.argv[4]
        corpus = convert_rcdata(directory, dataset, mode)
        with open(outfname, 'w') as outfile:
            json.dump(corpus, outfile, indent=2)
            print(" ...saved data to {0}.".format(outfname))
    else:
        print("""Usage:
    python3 {dataset} {mode} rc-data2jtr.py path/to/rc-data_dir save/to/rc-data.jtr.json
        where:
            {dataset} = {cnn, dailymail}
            {mode} = {train, dev, test}""")

if __name__ == '__main__':
    main()
