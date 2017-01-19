import json
import io
import random

def convert_scienceCloze_to_jtr(scienceQAFile):

    instances = []

    f = io.open(scienceQAFile, "r", encoding="utf-8")

    for l in f:
        l = l.strip().lower().split("\t")  # do the lower case preprocessing here
        try:
            quest, answs, cands, context, contextID = l
        except ValueError:
            print(l)
            continue

        context = context[2:-2].split('\', \'')

        support = []
        for i, c in enumerate(context):
            support.append({"id": contextID + "_" + str(i), "text": c})
        candidates = cands[2:-2].split('\', \'')

        qdict = {
            'question': quest,
            'candidates': [
                {
                    'text': cand
                } for cand in candidates
                ],
            'answers': [{'text': answs}]
        }
        qset_dict = {
            'support': support,
            'questions': [qdict]
        }

        instances.append(qset_dict)


    instances.append(qset_dict)
    random.shuffle(instances)

    corpus_dict = {
        'meta': "scienceQA.json",
        'instances': instances
    }

    f.close()

    return corpus_dict



if __name__ == "__main__":
    corpus = convert_scienceCloze_to_jtr("../data/scienceQA/clozeSummaryLocal_test.txt")
    with open("../data/scienceQA/scienceQA_clozeSummaryLocal_test.json", 'w') as outfile:
        json.dump(corpus, outfile, indent=2, ensure_ascii=False)