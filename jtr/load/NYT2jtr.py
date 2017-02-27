"""
This files allows creating a jtr train and test datafiles for the NYT corpus
based on ../data/NYT/naacl2013.txt
* train file: one jtr per train instance ; candidates = all entity pairs
* test file: one jtr per test relation, with all correct answers in the valid answers list

usage:
python3 NYT2jtr path/to/naacl2013.txt mode >  naacl2013_mode.jtr.json
in which mode = train or test
"""

import json

def load_naacl2013(path, mode):
    assert mode in ['train','test','Train','Test'], 'provide mode argument: train or test'
    facts = []
    #load facts of type mode
    with open(path) as fID:
        for line in fID:
            if len(line.strip())>0:
                r,e1,e2,typ,truth = line.strip().split('\t')
                rel = r
                truth = {'1.0':True,'0.0':False}[truth]
                tup = '(%s|||%s)'%(e1,e2)
                #if typ.lower()==mode.lower():
                facts.append(((rel,tup),truth,typ))

    #global candidates: all entity tuples appearing in the training data (for model F, test data should have no tuples not occurring in training data)
    tup_candidates_train = set([f[0][1] for f in facts if f[2].lower()=='train'])

    
    facts = [f for f in facts if f[0][1] in tup_candidates_train]
    
    trainmap = {}
    testmap = {}
    for f in facts:
        if f[2].lower()=='train':
            if f[0][0] not in trainmap:
                trainmap[f[0][0]]=set()
                testmap[f[0][0]]=set()
            trainmap[f[0][0]].add(f[0][1])
    
    
 
    for f in facts:
        if f[2].lower()=='test' and f[0][1] not in trainmap[f[0][0]]:
            testmap[f[0][0]].add(f[0][1])
    
    tup_candidates_test={}
    for rel in testmap:
        tup_candidates_test[rel]=[]
        for tup in testmap[rel]:
            tup_candidates_test[rel].append({'text':tup})


    if mode.lower()=='train':
        return create_train_jtr(facts,tup_candidates_train)
    elif mode.lower()=='test':
        return create_test_jtr(facts, tup_candidates_test)



def create_train_jtr(trainfacts,tuples):
    """one jtr per positive train fact"""
    instances = [{'support':[],
                  'questions':[{'question':fact[0][0],
                                'answers':[{'text':fact[0][1]}]
                                }]
                  }  for fact in trainfacts if fact[1] and fact[2].lower()=='train']#only add true facts for training (data should be consistent with that)
    #@Johannes: originally we had 'instances' as the jtr format; now: added metadata field 'meta' and 'globals' field with the overall candidates.
    return {'meta':'MFmodel-train',
            'globals':{'candidates':[{'text':tup} for tup in tuples]},
            'instances':instances
          }


def create_test_jtr(testfacts,tuples):
    """one jtr per positive test fact"""
    instances = [{'support':[],
                  'questions':[{'question':fact[0][0],
                                'answers':[{'text':fact[0][1]}],
                                'candidates':tuples[fact[0][0]]
                                }]
                  }  for fact in testfacts if fact[1] and fact[2].lower()=='test']#only add true facts for training (data should be consistent with that)
    #@Johannes: originally we had 'instances' as the jtr format; now: added metadata field 'meta' and 'globals' field with the overall candidates.
    return {'meta':'MFmodel-test',
            'instances':instances
          }



def main():
    import sys
    if len(sys.argv) == 3:
        data = load_naacl2013(sys.argv[1],sys.argv[2])
        print(json.dumps(data, indent=2))
    else:
        print("""usage: python3 NYT2jtr.py path/to/naacl2013.txt mode >  naacl2013_mode.jtr.json
              in which mode = train or test""")

if __name__ == "__main__":
    main()
