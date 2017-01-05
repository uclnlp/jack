import json
from collections import defaultdict


def main():
    # = parse_cbt_example(instances[0])
    import sys
    if len(sys.argv) == 2:
        with open(sys.argv[1], 'r') as f:
            sentihood_data = json.load(f)

        convert_to_quebap(sentihood_data)


def convert_to_quebap(sentihood_data, exhaustive=True):
    instances = []
    # collect all aspects
    aspects = set()
    for instance in sentihood_data:
        if 'opinions' in instance.keys():
            for opinion in instance['opinions']:
                aspects.add(opinion['aspect'])
    for instance in sentihood_data:
        text = instance['relevant_text']
        answers = defaultdict(lambda: 'Neutral')
        if 'opinions' in instance.keys():
            for opinion in instance['opinions']:
                aspect = opinion['aspect']
                answers[aspect] = opinion['sentiment']

        for aspect in aspects if exhaustive else answers.keys():
            reading_instance = {
                'support': [{'text': text}],
                'questions': [aspect],
                'answers': [{'text': answers[aspect]}]
            }
            instances.append(reading_instance)

    # print(json.dumps(sentihood_data, indent=2))
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
    print(json.dumps(reading_dataset, indent=2))


if __name__ == "__main__":
    main()
