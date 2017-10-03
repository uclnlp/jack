import json
import io


def create_snippet(file_path, first_n=5):
    with open(file_path, 'r') as f:
        return [next(f) for _ in range(first_n)]


def create_jtr_snippet(file_path):
    return convert_simplequestions(file_path, first_n=5)


def convert_simplequestions(file_path, first_n=None):
    instances = []
    f = io.open(file_path, "r")
    i = 0
    for l in f:
        i += 1
        if first_n and i > first_n:
            break
        subj, rel, obj, qu = l.strip().split("\t")

        support = [" ".join([subj, rel])]
        qdict = {
            'question': qu,
            'answers': [obj]
        }
        qset_dict = {
            'support': [{'text': supp} for supp in support],
            'questions': [qdict]
        }
        instances.append(qset_dict)

    corpus_dict = {
        'meta': "simpleQuestions.json",
        'instances': instances
    }

    f.close()

    return corpus_dict


def main():
    # some tests:
    # raw_data = load_cbt_file(path=None, part='valid', mode='NE')
    # instances = split_cbt(raw_data)
    # = parse_cbt_example(instances[0])

    import sys
    if len(sys.argv) == 3:
        # corpus = create_jtr_snippet(sys.argv[1])
        # out = create_snippet(sys.argv[1])
        # with open(sys.argv[2], 'w') as outfile:
        #     outfile.writelines(out)
        corpus = convert_simplequestions(sys.argv[1])
        with open(sys.argv[2], 'w') as outfile:
            json.dump(corpus, outfile, indent=2)
    else:
        print("Usage: python3 simpleQuestions2jtr.py path/to/simpleQuestions save/to/simpleQuestions.jack.json")


if __name__ == "__main__":
    main()
