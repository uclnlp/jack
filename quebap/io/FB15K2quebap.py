"""
Loader for the fb15k dataset.

METADATA:   Training data: 483142 triples (s,r,o)
            14951 different entities
            1345 different relation types
download_string = "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz"
webpage: https://everest.hds.utc.fr/doku.php?id=en:transe
paper reference:
@incollection{NIPS2013_5071,
    title = {Translating Embeddings for Modeling Multi-relational Data},
    author = {Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto and Weston, Jason and Yakhnenko, Oksana},
    booktitle = {Advances in Neural Information Processing Systems 26},
    editor = {C. J. C. Burges and L. Bottou and M. Welling and Z. Ghahramani and K. Q. Weinberger},
    pages = {2787--2795},
    year = {2013},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf}
}
"""

import json

# TODO memory efficiency: Don't store support explicitly!
size = 5000

def load_fb15k_triples(part='train'):
    """ loads the raw data from files provided. input parameter 'part' can be
    either 'train', 'valid' or 'test'.
    """
    path = '/Users/Johannes/PhD/kebab/pre/FB15k/'
    filename = path + 'freebase_mtr100_mte100-' + part + '.txt'
    with open(filename, 'r') as f:
        raw_lines = f.readlines()
    triples = [line.strip('\n').split('\t') for line in raw_lines]
    return triples


def extract_unique_entities_and_relations(triples, save_entities=False):
    """ Identify the sets of unique entities and relations in a set of triples.
    Return as ordered lists.
    """
    s_entities = set([triple[0] for triple in triples])
    o_entities = set([triple[2] for triple in triples])
    relations = sorted(list(set([triple[1] for triple in triples])))
    all_entities = sorted(list(s_entities | o_entities))
    if save_entities:
        D = {"data": all_entities}
        with open("entities_list.json", 'w') as f:
            json.dump(D, f)
    return all_entities, relations


def get_neighbourhood(triple, other_triples):
    """ obtain the neighbourhood for a single fact. This is, return a list of
    all facts that share either an entity or the relation with the fact.
    Inputs:
    - triple: the fact under consideration
    - other_triples: set of other triples from which to look for neighbours
    Note: Inefficient to use if applied on entire database """
    anchor_triple = set(triple)
    neighbourhood = [other_triple for other_triple in other_triples \
                    if len( anchor_triple & set(other_triple) ) > 0]
    return neighbourhood


def get_facts_per_entity(entities, triples, save=False):
    """ obtain a dictionary with all facts that contain an entity.
    Inputs:
        - entities: List of unique entities in triples
        - tiples: list of string triples
    """
    # might take (a few) minutes, looping over 15K entities.
    if not save:
        #print("loading entities...")
        with open("entities_neighbourhood.json", 'r') as f:
            D = json.load(f)
        return D
    else:
        D = {}
        for i,e in enumerate(entities):
            if not i%50: # monitoring progress
                #print(i)
                pass
            entity_support = [index for index,fact in enumerate(triples) if e in fact]
            D[e] = entity_support
        with open("entities_neighbourhood.json", 'w') as f:
            json.dump(D, f)
            #print("saving entities succesfully.")
    return D


def get_facts_per_relation(relations, triples, save=False):
    """ Same as get_facts_per_entity, but for relations
    """
    if not save:
        with open("relations_neighbourhood.json", 'r') as f:
            D = json.load(f)
            #print("loading relations succesfully.")
        return D
    else:
        D = {}
        for i,r in enumerate(relations):
            if not i%50: #monitoring progress
                #print(i)
                pass
            relation_support = [index for index,fact in enumerate(triples) if r in fact]
            D[r] = relation_support
        with open("relations_neighbourhood.json", 'w') as f:
            json.dump(D, f)
            #print("saving relations succesfully.")
    return D


def get_all_1_neighbourhoods(triples, entity_dict, relation_dict,
                            include_relations=False, save=True):
    """ extract neighbours for all facts in the KB.
    """
    if not save:
        with open("neighbourhood.json", 'r') as ff:
            #print(ff.name)
            neighbourhoods = json.load(ff)
            #print("loaded neighbourhoods succesfully.")
        return neighbourhoods
    else:
        neighbourhoods = []
        for i, triple in enumerate(triples[:size]):
            if not i%100: #monitoring progress
                #print(i)
                pass
            neighbours = entity_dict[triple[0]] + entity_dict[triple[2]]
            if include_relations:
                neighbours += relation_dict[triple[1]]
            # use unique neighbours, remove current triple, sort.
            if len(neighbours) == 0:
                neighbours = []
            else:
                neighbours = sorted(list(set(neighbours).difference(set([i]))))
            neighbourhoods.append(neighbours)
        with open("neighbourhood.json", 'w') as f:
            #print("saving neighbourhoods to file...")
            D = {"data": neighbourhoods}
            json.dump(D, f)
            #print("saving neighbourhoods succesfully.")
        return neighbourhoods


def convert_triple_to_text(triple):
    s,r,o = triple
    return s + " " + r + " " + o + "."


def parse_fb15k_question(triple):
    # TODO make 1:n queries possible here.
    subject, relation, obj = triple
    questions = []
    for i in range(0,1): #single question
        qdict = {}
        candidates = [{'text' : "filename: datasets/fb15k/entities_list.json"} ]
        answer = {'text': obj}
        qdict  = {
            "question" : relation + " " + subject + "?",
            "candidates" : candidates,
            "answers": [answer]
        }
        questions.append(qdict)
    return questions


def convert_instance(triple, triples, neighbours):
    support_text = [convert_triple_to_text(triples[fact]) for fact in \
            neighbours if len(neighbours)>0]
    qset_dict = {}
    qset_dict['support'] = [ {'text': " ".join(support_text)} ]
    qset_dict['questions'] = parse_fb15k_question(triple)
    return qset_dict


def convert_fb15k(triples, neighbourhoods):
    """ target format:
    "question": "born_in(BarackObama, ? )",
    "support": [
      "BarackObama was born in Hawaii",
      "president_of(BarackObama, USA)"
    ]
    "candidates": { "filename": "filename for file with list of candidates" }
    """
    corpus = []
    for i_triple, (trip, nhbrs) in enumerate(zip(triples[:size], neighbourhoods[:size])):
        if not i_triple%100:
            #print(i_triple)
            pass
        corpus.append(convert_instance(trip, triples, nhbrs) )
    return corpus


if __name__ == "__main__":
    triples = load_fb15k_triples(part='train')
    entities, relations = extract_unique_entities_and_relations(triples)
    _ = get_neighbourhood(triples[0], triples)
    entity_dict = get_facts_per_entity(entities, triples)
    relation_dict = get_facts_per_relation(relations, triples)
    neighbourhoods = get_all_1_neighbourhoods(triples, entity_dict, relation_dict)
    corpus = convert_fb15k(triples, neighbourhoods)
    print(json.dumps(corpus, indent=2))
