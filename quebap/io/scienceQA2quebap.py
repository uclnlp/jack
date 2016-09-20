import json
import io
import random

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
        try:
            quest, answs, cands, context, contextID = l
        except ValueError:
            print(l)
            continue

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
        'answers': [{'text': ans} for ans in answers_last_split]
    }
    qset_dict = {
        'support': support,
        'questions': [qdict]
    }

    instances.append(qset_dict)
    random.shuffle(instances)

    corpus_dict = {
        'meta': "scienceQA.json",
        'instances': instances
    }

    f.close()

    return corpus_dict



def convert_scienceQACloze_to_quebap(scienceQAFile):

    instances = []

    f = io.open(scienceQAFile, "r")

    question_last = ""
    answers_last = ""
    answers_last_split = [] # here, the answers are not synonyms, but stand for clozent1 and clozent2, respectively
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
                #question_last
                #question_last_all = [question_last.replace("clozent0", answers_last_split[0]), question_last.replace("clozent1", answers_last_split[1])]
                question_last_all = [question_last + "(clozent0: " + answers_last_split[0] + ", clozent1: ?)",
                                     question_last + "(clozent1: " + answers_last_split[1] + ", clozent0: ?)"]
                for i, question_rep in enumerate(question_last_all):
                    qdict = {
                        'question': question_rep,
                        'candidates': [
                            {
                                'text': cand
                            } for cand in candidates
                            ],
                        'answers': [{'text': answers_last_split[i]}]
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
    #question_last_all = [question_last.replace("clozent0", answers_last_split[0]),
    #                     question_last.replace("clozent1", answers_last_split[1])]
    question_last_all = [question_last + "(clozent0: " + answers_last_split[0] + ", clozent1: ?)",
                         question_last + "(clozent1: " + answers_last_split[1] + ", clozent0: ?)"]
    for i, question_rep in enumerate(question_last_all):
        qdict = {
            'question': question_rep,
            'candidates': [
                {
                    'text': cand
                } for cand in candidates
                ],
            'answers': [{'text': answers_last_split[i]}]
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



def main(mode):
    # some tests:
    # raw_data = load_cbt_file(path=None, part='valid', mode='NE')
    # instances = split_cbt(raw_data)
    # = parse_cbt_example(instances[0])
    if mode == "KBP":
        corpus = convert_scienceQA_to_quebap("../../quebap/data/scienceQA/scienceIE_kbp_all.txt")#scienceQA_KBP_all.txt")#scienceQA.txt")
        with open("../../quebap/data/scienceQA/scienceQA_kbp_all.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        outfile.close()
    elif mode == "Cloze":
        corpus = convert_scienceQACloze_to_quebap("../../quebap/data/scienceQA/scienceQA_cloze_small.txt")
        with open("../../quebap/data/scienceQA/scienceQA_cloze.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        outfile.close()

if __name__ == "__main__":
    main("KBP")