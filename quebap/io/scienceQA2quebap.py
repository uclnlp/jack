import json
import io
import random
from gensim.models import Phrases
from nltk import word_tokenize, pos_tag, sent_tokenize
import copy

def convert_scienceQA_to_quebap(scienceQAFile, addSupport=True):

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
            if addSupport == True:
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
            if addSupport == True:
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



def convert_scienceQACloze_to_quebap(scienceQAFile, addSupport=False, shortsupport=True):

    instances = []

    f = io.open(scienceQAFile, "r", encoding="utf-8")

    question_last = ""
    answers_last = ""
    answers_last_split = [] # here, the answers are not synonyms, but stand for clozent1 and clozent2, respectively
    support = []
    candidates = []

    #i = 0

    for l in f:
        #i += 1
        #if i > 200:
        #    break
        l = l.strip().lower().split("\t")  # do the lower case preprocessing here
        try:
            quest, answs, cands, context, contextID = l
            if shortsupport == True:
                add_s = False
                for c in cands[2:-2].split('\', \''):
                    if c in answs[2:-2].split('\', \''):
                        add_s = True
                        break
                if add_s == False:
                    continue
        except ValueError:
            print(l)
            continue

        # append
        if quest == question_last and answs == answers_last:
            if shortsupport == True:
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

            candidates = []
            support = []
            answers_last_split = []

            question_last = quest
            answers_last = answs
            answers_last_split = answs[2:-2].split('\', \'')

            if addSupport == True:
                support.append({"id": contextID, "text": context})
            candidates.extend(cands[2:-2].split('\', \''))


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



def buildAuthrKDictNew():
    authork = open("/Users/Isabelle/Documents/UCLMR/publication-extract/data/_paper_keyphrases_author_filtered.txt", "r")  # _paper_keyphrases_author.txt # should use author_train, need to do a new split of data
    authorkmap = {}

    for a in authork:
        # print(a)
        aspl = a.strip("\n").lower().split("\t")
        if len(aspl) < 2:
            continue
        k = aspl[0].replace(" ", "_")
        try:
            cnt = int(aspl[1])
        except ValueError:
            continue
        if not k in authorkmap:
            try:
                authorkmap[k] = cnt
            except ValueError:
                continue
        else:
            currcnt = authorkmap[k]
            authorkmap[k] = int(currcnt) + cnt

    return authorkmap


def checkCandsSing(authorkmap, toks_lower, pos):
    trans_sent = transSent(
        transSent(transSent(transSent(toks_lower, authorkmap), authorkmap), authorkmap), authorkmap)
    authcands, _ = getCandidatesAuthorDefToks(authorkmap, trans_sent)  # that's the testing sentence

    authcands = checkCandidates(authcands, pos)

    authcands_new = []
    for cand in authcands:
        t = cand.replace("_", " ")
        if " " in t:
            authcands_new.append(t)

    return authcands_new


def checkCandidates(toks, pos):
    phr = []
    for t in toks:
        tpos = False  # suitable or not
        if " " in t:
            tlast = t.split(" ")[-1]
            poscnt = 0
            for poss in pos:
                poscnt += 1
                if poss[0].lower() == tlast:
                    if poss[1].startswith("N"):# or poss[1].startswith("V"):
                        tpos = True
                    break
            if tpos == True:
                phr.append(t.replace("_", " "))
        else:
            poscnt = 0
            for poss in pos:
                poscnt += 1
                if poss[0].lower() == t:
                    if poss[1].startswith("N"): #or poss[1].startswith("V"):
                        tpos = True
                    break
            #pos = pos[poscnt:]  # remove the seen ones
            if tpos == True:
                phr.append(t.replace("_", " "))
    return phr



def getCandidatesAuthorDefToks(keymap, text):
    phr = []
    phrnorm = set()
    for tok in text:
        # add MWE automatically
        #if "_" in tok:
        #    phr.append(tok.lower())
        #    phrnorm.add(tok.lower())
        if tok in keymap:
            phr.append(tok.replace("_", " "))
            phrnorm.add(tok.lower().replace("_", " "))

    return phr, phrnorm




def transSent(s, vocab):
    mincnt = 5
    last_bigram = False
    delimiter = "_"
    new_s = []
    for word_a, word_b in zip(s, s[1:]):
        bigram_word = delimiter.join((word_a, word_b))
        if bigram_word in vocab and vocab[bigram_word] >= mincnt and not last_bigram:
            new_s.append(bigram_word)
            last_bigram = True
            continue

        if not last_bigram:
            new_s.append(word_a)
        last_bigram = False

    if s:  # add last word skipped by previous loop
        last_token = s[-1]
        if not last_bigram:
            new_s.append(last_token)

    return new_s


def transSentRepl(s, word_check, repl):
    s = s.replace(" " + word_check + " ", " " + repl + " ").replace(" " + word_check + ".", " " + repl + ".")
    if s.startswith(word_check):
        s = s.replace(word_check + " ", repl + " ", 1)
    return s



def convert_scienceQACloze_withsupport_to_quebap(scienceQAFile, addSupport=True):

    # get rid of duplicate cands and sents, aggregate over multiple lines (possibly), permute for each cloze Q or support if they're used as
    # support or as Q if they contain the answer


    authorkmap = buildAuthrKDictNew()

    phrmodel = Phrases.load("/home/isabelle/publication-extract/models_out/phrase_all_five_big.model")#phrase_all_five_small.model")
    phrmodelvocab = phrmodel.vocab
    vocab = authorkmap.copy()
    vocab.update(phrmodelvocab)
    authorkmap = vocab

    instances = []

    f = io.open(scienceQAFile, "r", encoding="utf-8")

    question_last = ""
    alternative_questions = []
    answers_last = ""
    answers_last_split = [] # here, the answers are not synonyms, but stand for clozent1 and clozent2, respectively
    support = []
    candidates = []

    #i = 0

    for l in f:
        #i += 1
        #if i > 100:
        #    break
        l = l.strip().lower().split("\t")  # do the lower case preprocessing here
        try:
            quest, answs, cands, cont, contextID = l
            #print(quest, cont)

            if not quest.endswith(" ."):
                quest = quest + " ."
        except ValueError:
            print(l)
            continue

        # append
        if quest == question_last and answs == answers_last:
            cont_tmp = cont[2:-2].split('\', \'')
            context = list(set(cont_tmp))

            addCands = True

            for ii, c in enumerate(context):
                toks_lower = word_tokenize(c)
                c = " ".join(toks_lower)
                if not c.endswith(" ."):
                    c = c + " ."

                # added this
                for a in answers_last_split:
                    if a in c:
                        c_anon = transSentRepl(" ".join(word_tokenize(c)), a, "xxxxx")

                if quest == c_anon:
                    print("Support same as question!")
                    addCands = False
                    question_last, answers_last = "", ""
                    continue

                support.append({"id": contextID + "_" + str(ii), "text": c})
                pos = pos_tag(toks_lower)
                authcands_ks = checkCandsSing(authorkmap, toks_lower, pos)
                add_c = False
                for k in authcands_ks:
                    candidates.append(k)
                    if k in answers_last_split:
                        add_c = True
                if add_c == True:
                    alternative_questions.append(c)

            if addCands == True:
                candidates.extend(cands[2:-2].split('\', \''))

        else:
            # then this is the first iteration
            if question_last != "":
                candidates = list(set(candidates)) # remove duplicates
                #question_last
                #question_last_all = [question_last.replace("clozent0", answers_last_split[0]), question_last.replace("clozent1", answers_last_split[1])]
                #question_last_all = [question_last + "(clozent0: " + answers_last_split[0] + ", clozent1: ?)",
                #                     question_last + "(clozent1: " + answers_last_split[1] + ", clozent0: ?)"]

                if len(support) == 0:
                    continue

                qdict = {
                    'question': question_last,
                    'candidates': [
                        {
                            'text': cand
                        } for cand in candidates
                        ],
                    'answers': [
                        {
                            'text': a
                        } for a in answers_last_split
                        ],
                }
                qset_dict = {
                    'support': support,
                    'questions': [qdict]
                }

                instances.append(qset_dict)


                for q in alternative_questions:
                    for a in answers_last_split:
                        if a in q:
                            cloze_s0 = transSentRepl(" ".join(word_tokenize(q)), a, "xxxxx")
                    support_copy = copy.deepcopy(support)
                    for s in support:
                        if s["text"] == q:
                            support_copy.remove(s)

                    a_trans = question_last.replace("xxxxx", answers_last_split[0])
                    support_copy.append({"id": "highlight", "text": a_trans})

                    qdict = {
                        'question': cloze_s0,
                        'candidates': [
                            {
                                'text': cand
                            } for cand in candidates
                            ],
                        'answers': [
                            {
                                'text': a
                            } for a in answers_last_split
                            ],
                    }
                    qset_dict = {
                        'support': support_copy,
                        'questions': [qdict]
                    }

                    instances.append(qset_dict)


            candidates = []
            support = []
            answers_last_split = []
            alternative_questions = []

            question_last = quest
            answers_last = answs
            #answers_last_split = answs[2:-2].split('\', \'')

            if answs.startswith("["):
                answers_last_split = answs[2:-2].split('\', \'')
            else:
                answers_last_split = answs.split('\', \'')

            addCands = True

            if addSupport == True:
                cont_tmp = cont[2:-2].split('\', \'')
                context = list(set(cont_tmp))

                for ii, c in enumerate(context):

                    toks_lower = word_tokenize(c)
                    c = " ".join(toks_lower)   # added this!!!
                    if not c.endswith(" ."):
                        c = c + " ."

                    # added this
                    for a in answers_last_split:
                        if a in c:
                            c_anon = transSentRepl(" ".join(word_tokenize(c)), a, "xxxxx")

                    if quest == c_anon:
                        print("Support same as question!")
                        addCands = False
                        question_last, answers_last = "", ""
                        continue

                    support.append({"id": contextID + "_" + str(ii), "text": c})
                    pos = pos_tag(toks_lower)
                    authcands_ks = checkCandsSing(authorkmap, toks_lower, pos)
                    add_c = False
                    for k in authcands_ks:
                        candidates.append(k)
                        if k in answers_last_split:
                            add_c = True
                    if add_c == True:
                        alternative_questions.append(c)

            if len(support) != len(alternative_questions):
                print("more support than alt questions")

            if addCands == True:
                candidates.extend(cands[2:-2].split('\', \''))


    candidates = list(set(candidates))  # remove duplicates

    qdict = {
        'question': question_last,
        'candidates': [
            {
                'text': cand
            } for cand in candidates
            ],
        'answers': [
            {
                'text': a
            } for a in answers_last_split
            ],
    }
    qset_dict = {
        'support': support,
        'questions': [qdict]
    }

    if len(support) != 0:
        instances.append(qset_dict)

    for q in alternative_questions:

        if len(support) == 0:
            continue

        for a in answers_last_split:
            if a in q:
                cloze_s0 = transSentRepl(" ".join(word_tokenize(q)), a, "xxxxx")
        support_copy = copy.deepcopy(support)
        for s in support:
            if s["text"] == q:
                support_copy.remove(s)

        a_trans = question_last.replace("xxxxx", answers_last_split[0])
        support_copy.append({"id": "highlight", "text": a_trans})

        qdict = {
            'question': cloze_s0,
            'candidates': [
                {
                    'text': cand
                } for cand in candidates
                ],
            'answers': [
                {
                    'text': a
                } for a in answers_last_split
                ],
        }
        qset_dict = {
            'support': support_copy,
            'questions': [qdict]
        }

        instances.append(qset_dict)


    corpus_dict = {
        'meta': "scienceQA.json",
        'instances': instances
    }

    f.close()

    return corpus_dict




def convert_scienceQAComb_withsupport_to_quebap(scienceQAFile, addSupport=True):

    # get rid of duplicate cands and sents, aggregate over multiple lines (possibly), permute for each cloze Q or support if they're used as
    # support or as Q if they contain the answer


    authorkmap = buildAuthrKDictNew()

    """phrmodel = Phrases.load("/home/isabelle/publication-extract/models_out/phrase_all_five_big.model")#phrase_all_five_big.model")#phrase_all_five_small.model")
    phrmodelvocab = phrmodel.vocab
    vocab = authorkmap.copy()
    vocab.update(phrmodelvocab)
    authorkmap = vocab"""

    instances = []

    f = io.open(scienceQAFile, "r", encoding="utf-8")

    question_last = ""
    rel_last = ""
    alternative_questions = []
    answers_last = ""
    answers_last_split = [] # here, the answers are not synonyms, but stand for clozent1 and clozent2, respectively
    support = []
    candidates = []

    i = 0

    for l in f:
        i += 1
        if i > 20:
            break
        l = l.strip().lower().split("\t")  # do the lower case preprocessing here
        try:
            quest, kbrel, answs, cands, cont, contextID = l

            if not quest.endswith(" ."):
                quest = quest + " ."

            if answs.startswith("["):
                answers_spl = answs[2:-2].split('\', \'')
            else:
                answers_spl = answs.split('\', \'')
            if cands.startswith("["):
                cands_spl = cands[2:-2].split('\', \'')
            else:
                cands_spl = cands.split('\', \'')

            if kbrel != "none" and kbrel != "unrelated":
                for cand in cands_spl:
                    if not cand in answers_spl:
                        kbrel = cand + " | " + kbrel
                    break

        except ValueError:
            print(l)
            continue

        # append
        if quest == question_last and answs == answers_last:
            cont_tmp = cont[2:-2].split('\', \'')
            context = list(set(cont_tmp))

            addCands = True

            for ii, c in enumerate(context):
                toks_lower = word_tokenize(c)

                c = " ".join(toks_lower)
                if not c.endswith(" ."):
                    c = c + " ."

                for a in answers_last_split:
                    if a in c:
                        c_anon = transSentRepl(" ".join(word_tokenize(c)), a, "xxxxx")

                if quest == c_anon:
                    print("Support same as question!")
                    addCands = False
                    question_last, answers_last = "", ""
                    continue

                support.append({"id": contextID + "_" + str(ii), "text": c})

                pos = pos_tag(toks_lower)
                authcands_ks = checkCandsSing(authorkmap, toks_lower, pos)
                add_c = False
                for k in authcands_ks:
                    candidates.append(k)
                    if k in answers_last_split:
                        add_c = True
                if add_c == True:
                    alternative_questions.append(c)

            if addCands == True:
                candidates.extend(cands[2:-2].split('\', \''))

        else:
            # then this is the first iteration
            if question_last != "":
                if len(support) == 0:
                    continue

                candidates = list(set(candidates)) # remove duplicates

                qdict = {
                    'question':  question_last, #rel_last, # question_last,  # ALTERNATIVE here
                    'candidates': [
                        {
                            'text': cand
                        } for cand in candidates
                        ],
                    'answers': [
                        {
                            'text': a
                        } for a in answers_last_split
                        ],
                }
                qset_dict = {
                    'support': support,
                    'questions': [qdict]
                }

                instances.append(qset_dict)


                for q in alternative_questions:
                    for a in answers_last_split:
                        if a in q:
                            cloze_s0 = transSentRepl(" ".join(word_tokenize(q)), a, "xxxxx")
                    support_copy = copy.deepcopy(support)
                    for s in support:
                        if s["text"] == q:
                            support_copy.remove(s)

                    a_trans = question_last.replace("xxxxx", answers_last_split[0])
                    support_copy.append({"id": contextID + "_" + str(ii) + "highlight", "text": a_trans})

                    qdict = {
                        'question': cloze_s0,#rel_last, #cloze_s0,
                        'candidates': [
                            {
                                'text': cand
                            } for cand in candidates
                            ],
                        'answers': [
                            {
                                'text': a
                            } for a in answers_last_split
                            ],
                    }
                    qset_dict = {
                        'support': support_copy,
                        'questions': [qdict]
                    }

                    instances.append(qset_dict)


            candidates = []
            support = []
            answers_last_split = []
            alternative_questions = []

            question_last = quest
            answers_last = answs
            rel_last = kbrel
            #answers_last_split = answs[2:-2].split('\', \'')

            if answs.startswith("["):
                answers_last_split = answs[2:-2].split('\', \'')
            else:
                answers_last_split = answs.split('\', \'')

            addCands = True
            if addSupport == True:
                cont_tmp = cont[2:-2].split('\', \'')
                context = list(set(cont_tmp))

                for ii, c in enumerate(context):

                    toks_lower = word_tokenize(c)
                    c = " ".join(toks_lower)  # added this!!!
                    if not c.endswith(" ."):
                        c = c + " ."

                    # added this
                    for a in answers_last_split:
                        if a in c:
                            c_anon = transSentRepl(" ".join(word_tokenize(c)), a, "xxxxx")

                    if quest == c_anon:
                        print("Support same as question!")
                        addCands = False
                        question_last, answers_last = "", ""
                        continue

                    support.append({"id": contextID + "_" + str(ii), "text": c})
                    pos = pos_tag(toks_lower)
                    authcands_ks = checkCandsSing(authorkmap, toks_lower, pos)
                    add_c = False
                    for k in authcands_ks:
                        candidates.append(k)
                        if k in answers_last_split:
                            add_c = True
                    if add_c == True:
                        alternative_questions.append(c)

            if len(support) != len(alternative_questions):
                print("more support than alt questions")

            if addCands == True:
                candidates.extend(cands[2:-2].split('\', \''))


    candidates = list(set(candidates))  # remove duplicates

    qdict = {
        'question': question_last, #rel_last, #question_last,
        'candidates': [
            {
                'text': cand
            } for cand in candidates
            ],
        'answers': [
            {
                'text': a
            } for a in answers_last_split
            ],
    }
    qset_dict = {
        'support': support,
        'questions': [qdict]
    }

    if len(support) != 0:
        instances.append(qset_dict)

    for q in alternative_questions:

        if len(support) == 0:
            continue

        for a in answers_last_split:
            if a in q:
                cloze_s0 = transSentRepl(" ".join(word_tokenize(q)), a, "xxxxx")
        support_copy = copy.deepcopy(support)
        for s in support:
            if s["text"] == q:
                support_copy.remove(s)

        a_trans = question_last.replace("xxxxx", answers_last_split[0])
        support_copy.append({"id": contextID + "_" + str(ii) + "highlight", "text": a_trans})

        qdict = {
            'question': cloze_s0, #rel_last, #cloze_s0,
            'candidates': [
                {
                    'text': cand
                } for cand in candidates
                ],
            'answers': [
                {
                    'text': a
                } for a in answers_last_split
                ],
        }
        qset_dict = {
            'support': support_copy,
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
        corpus = convert_scienceQA_to_quebap("../../quebap/data/scienceQA/scienceIE_kbp_all.txt", addSupport=False)#scienceQA_KBP_all.txt")#scienceQA.txt")
        with open("../../quebap/data/scienceQA/scienceQA_kbp_all_nosupport.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        outfile.close()
    elif mode == "Cloze":
        #corpus = convert_scienceQACloze_to_quebap("../../quebap/data/scienceQA/_cloze_sept6_2016.txt", addSupport=True, shortsupport=True)
        corpus = convert_scienceQACloze_withsupport_to_quebap("../../quebap/data/scienceQA/cloze_with_support_2016-12-6_sorted.txt")#cloze_withcontext_sorted.txt")
        with open("../../quebap/data/scienceQA/scienceQA_cloze_withcont_2016-12-6_small.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        outfile.close()

    elif mode == "Comb":
        # corpus = convert_scienceQACloze_to_quebap("../../quebap/data/scienceQA/_cloze_sept6_2016.txt", addSupport=True, shortsupport=True)
        corpus = convert_scienceQAComb_withsupport_to_quebap(
            "../../quebap/data/scienceQA/kbp_with_support_2016-11-28.txt")  # cloze_withcontext_sorted.txt")
        with open("../../quebap/data/scienceQA/scienceQA_comb_cloze_withcont_2016-11.json", 'w') as outfile:
            json.dump(corpus, outfile, indent=2)

        outfile.close()


if __name__ == "__main__":
    main("Comb")
