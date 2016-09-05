import json
import io

def convert_scienceQA_to_quebap(scienceQAFile):

    instances = []

    f = io.open(scienceQAFile, "r")

    question_last = ""
    answers_last = ""
    answers_last_split = []
    support = []
    candidates = []

    for l in f:
        l = l.strip().lower().split("\t")  # do the lower case preprocessing here
        quest, answs, cands, context, contextID = l

        # append
        if quest == question_last and answs == answers_last:
            support.append({"id": contextID, "text": context})
            candidates.extend(cands[2:-2].split('\', \''))

        else:
            # then this is the first iteration
            if question_last != "":
                candidates = list(set(candidates)) # remove duplicates
                qdict = {
                    'question': question_last,
                    'candidates': [
                        {
                            'text': cand
                        } for cand in candidates
                        ],
                    'answers': [{'text': ans} for ans in answers_last_split]
                }
                qset_dict = {
                    'support': support,
                    'questions': [qdict]
                }

                instances.append(qset_dict)

            question_last = quest
            answers_last = answs
            answers_last_split = answs[2:-2].split('\', \'')
            support = [{"id": contextID, "text": context}]
            candidates = cands[2:-2].split('\', \'')


    candidates = list(set(candidates))  # remove duplicates
    qdict = {
        'question': question_last,
        'candidates': [
            {
                'text': cand
            } for cand in candidates
            ],
        'answers': answers_last_split
    }
    qset_dict = {
        'support': support,
        'questions': [qdict]
    }

    instances.append(qset_dict)


    corpus_dict = {
        'meta': "scienceQA.json",
        'instances': instances
    }

    f.close()

    return corpus_dict


def main():
    # some tests:
    # raw_data = load_cbt_file(path=None, part='valid', mode='NE')
    # instances = split_cbt(raw_data)
    # = parse_cbt_example(instances[0])
    corpus = convert_scienceQA_to_quebap("../../quebap/data/scienceQA/scienceQA.txt")
    with open("../../quebap/data/scienceQA/scienceQA.json", 'w') as outfile:
        json.dump(corpus, outfile, indent=2)

    outfile.close()

if __name__ == "__main__":
    main()