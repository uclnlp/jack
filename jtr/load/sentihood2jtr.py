import json
from collections import defaultdict
import sys
import os


def main():
    # = parse_cbt_example(instances[0])
    if len(sys.argv) == 2:
        with open(sys.argv[1], 'r') as f:
            sentihood_data = json.load(f)

        convert_to_jtr(sentihood_data)
    elif len(sys.argv) ==1:
        data_path = '../data/sentihood/'
        filenames = ['sentihood-train.json', 'sentihood-dev.json',
        'sentihood-test.json']
        for i, f in enumerate(filenames):
            raw_data = json.load(open(os.path.join(data_path, f)))
            instances = convert_to_jtr(raw_data)

            if i == 0: # training data -> write overfit set
                json.dump(wrap_into_jtr_global(instances[:100]),
                        open('../../tests/test_data/sentihood/overfit.json','w'),
                        indent=2)

            # write data sets for smalldata tests
            json.dump(wrap_into_jtr_global(instances[:1000]),
                    open(os.path.join('../../tests/test_data/sentihood/',f),'w'),
                    indent=2)

def wrap_into_jtr_global(instances):
    reading_dataset = {
        'globals': {
            'candidates': [
                {'text': 'Negative'},
                {'text': 'Positive'},
                {'text': 'Neutral'}
            ]
        },
        'instances': instances
    }
    return reading_dataset



def convert_to_jtr(sentihood_data, exhaustive=True):
    instances = []
    # collect all aspects
    aspects = set()
    for instance in sentihood_data:
        if 'opinions' in instance.keys():
            for opinion in instance['opinions']:
                aspects.add(opinion['aspect'])
    for instance in sentihood_data:
        text = instance['text']
        answers = defaultdict(lambda: 'Neutral')
        if 'opinions' in instance.keys():
            for opinion in instance['opinions']:
                aspect = opinion['aspect']
                answers[aspect] = opinion['sentiment']

        for aspect in aspects if exhaustive else answers.keys():
            reading_instance = {
                'support': [{'text': text}],
                'questions': [{'question': aspect, 'answers': [{'text': answers[aspect]}]}]
            }
            instances.append(reading_instance)

    return instances


if __name__ == "__main__":
    main()
