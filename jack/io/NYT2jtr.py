"""
This files allows creating a jtr train and test datafiles for the NYT corpus
based on jtr/data/NYT/naacl2013.txt
* train file: one jtr per train instance ; candidates = all entity pairs
* test file: one jtr per test relation, with all correct answers in the valid answers list

usage:
python3 NYT2jtr path/to/naacl2013.txt mode >  naacl2013_mode.jtr.json
in which mode = train or test
"""

import json


def load_naacl2013(path, mode):
    assert mode in ['train', 'test', 'Train', 'Test'], 'provide mode argument: train or test'
    facts = []
    # load facts of type mode
    with open(path) as fID:
        for line in fID:
            if len(line.strip()) > 0:
                r, e1, e2, typ, truth = line.strip().split('\t')
                rel = r
                truth = {'1.0': True, '0.0': False}[truth]
                tup = '(%s|||%s)' % (e1, e2)
                if typ.lower() == mode.lower():
                    facts.append(((rel, tup), truth, typ))

    # global candidates: all entity tuples appearing in the training data
    # (for model F, test data should have no tuples not occurring in training data)
    tup_candidates_train = sorted(list(set([f[0][1] for f in facts if f[2].lower() == 'train'])))
    tup_candidates_test = sorted(list(set([f[0][1] for f in facts if f[2].lower() == 'test'])))

    if mode.lower() == 'train':
        return create_train_jtr(facts, tup_candidates_train)
    elif mode.lower() == 'test':
        return create_test_jtr(facts, tup_candidates_test)


def create_train_jtr(trainfacts, tuples):
    """
    TODO comment
    one jack per positive train fact

    Args:
        trainfacts:
        tuples:

    Returns:

    """

    instances = [{
                     'support': [],
                     'questions': [{
                         'question': fact[0][0],
                         'candidates': '#/globals/candidates',
                         'answers': [{
                             'text': fact[0][1]
                         }]
                     }]
                     # only add true facts for training (data should be consistent with that)
                  } for fact in trainfacts if fact[1]]

    # @Johannes: originally we had 'instances' as the jack format;
    # now: added metadata field 'meta' and 'globals' field with the overall candidates.
    return {
        'meta': 'MFmodel-train',
        'globals': {
            'candidates': [{'text': tup} for tup in tuples]},
        'instances': instances
    }


def create_test_jtr(testfacts, tuples):
    """
    TODO comment
    one jack with all correct answers per relation in the test data

    Args:
        testfacts:
        tuples:

    Returns:

    """
    # map test relations to all true/false entity tuples
    relmap = {}
    for fact in testfacts:
        rel, tup = fact[0]
        truth = fact[1]
        if not truth:
            continue
        if rel not in relmap:
            relmap[rel] = {
                'candidates': set(),
                'answers': set()
            }

        # relmap[rel]['candidates'].add(tup)
        relmap[rel]['answers'].add(tup)

    # test instances:
    instances = [{
                     'support': [],
                     'questions': [{
                         'question': rel,
                         'candidates': [{'text': c} for c in relmap[rel]['candidates']],
                         'answers': [{'text': a} for a in relmap[rel]['answers']]
                     }]
                  } for rel in relmap]
    return {
        'meta': 'MFmodel-test',
        'globals': {
            'candidates': [{'text': tup} for tup in tuples]},
        'instances': instances
          }


def main():
    import sys
    if len(sys.argv) == 4:
        data = load_naacl2013(sys.argv[1], sys.argv[2])
        with open(sys.argv[3], 'w') as outfile:
            json.dumps(data, outfile, indent=2)

    else:
        print("""Usage:
    python3 NYT2jtr.py path/to/naacl2013 {mode} /save/to/naacl2013_mode.jack.json
        where {mode} = {train, test}""")

if __name__ == "__main__":
    main()
