"""
This files allows creating a quebap train and test datafiles for the NYT corpus
based on ../data/NYT/naacl2013.txt
* train file: one quebap per train instance ; candidates = all entity pairs
* test file: one quebap per test relation, with all correct answers in the valid answers list

usage:
python3 NYT2quebap path/to/naacl2013.txt mode >  naacl2013_mode.quebap.json
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
                if typ.lower()==mode.lower():
                    facts.append(((rel,tup),truth,typ))

    #global candidates: all entity tuples appearing in the training data (for model F, test data should have no tuples not occurring in training data) 
    tup_candidates = sorted(list(set([f[0][1] for f in facts if typ.lower()=='train'])))

    if mode.lower()=='train':
        return create_train_quebap(facts,tup_candidates)
    elif mode.lower()=='test':
        return create_test_quebap(facts)
    

    
def create_train_quebap(trainfacts,tuples):
    """one quebap per positive train fact"""
    instances = [{'support':[],
                  'questions':[{'question':fact[0][0],
                                'candidates':'#/globals/candidates',
                                'answers':[{'text':fact[0][1]}]
                                }] 
                  }  for fact in trainfacts if fact[1]]#only add true facts for training (data should be consistent with that)
    #@Johannes: originally we had 'instances' as the quebap format; now: added metadata field 'meta' and 'globals' field with the overall candidates.
    return {'meta':'MFmodel-train',
            'globals':{'candidates':[{'text':tup} for tup in tuples]},
            'instances':instances       
          }  

    
def create_test_quebap(testfacts):
    """one quebap with all correct answers per relation in the test data"""
    #map test relations to all true/false entity tuples
    relmap = {}
    for fact in testfacts: 
        rel,tup = fact[0]
        truth = fact[1]
        if not rel in relmap:
            relmap[rel]={'candidates':set(),'answers':set()}
        relmap[rel]['candidates'].add(tup)
        if truth:
            relmap[rel]['answers'].add(tup)
    #test instances: 
    instances = [{'support':[],
                  'questions':[{'question':rel,
                               'candidates':[{'text':c} for c in relmap[rel]['candidates']],
                               'answers':[{'text':a} for a in relmap[rel]['answers']]
                               }]
                  } for rel in relmap]
    return {'meta':'MFmodel-test',
            'globals':{},
            'instances':instances       
          }  



def main():
    import sys
    if len(sys.argv) == 3:
        data = load_naacl2013(sys.argv[1],sys.argv[2])
        print(json.dumps(data, indent=2))
    else:
        print("""usage: python3 NYT2quebap path/to/naacl2013.txt mode >  naacl2013_mode.quebap.json
              in which mode = train or test""")

if __name__ == "__main__":
    main()
