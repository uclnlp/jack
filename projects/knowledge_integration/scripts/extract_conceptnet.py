"""Extract ConceptNet assertions and add them to an assertion store."""

import gzip
import re

import spacy

from projects.knowledge_integration.knowledge_store import KnowledgeStore


def uncamel(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()


def only_string(concept_str):
    if concept_str.startswith("/r/"):
        return uncamel(concept_str[3:])
    else:
        res = concept_str[6:]
        if "/" in res:
            res = res[:res.index("/")]
        return res.replace("_", " ")


def normalize_and_sois(s):
    i = s.find("[[")
    offset = 0
    sois = list()
    while i >= 0:
        start = i - offset
        offset += 2
        sois.append((start, s.find("]]", i) - offset))
        offset += 2
        i = s.find("[[", i + 1)
    s_norm = s.replace("[[", "").replace("]]", "").lower()
    return s_norm, sois


__reg = r'^[^\t]+/en/([^\t])+/en/'  # only get connections between english concepts
def is_valid(l):
    return re.match(__reg, l) is not None and "/d/verbosity" not in l and (
        "/d/conceptnet/4/en" not in l or l.count("/s/contributor/omcs/") > 1)


def lemmatized(tokens, start, end):
    return " ".join(t.lemma_ for t in tokens if t.idx < end and t.idx >= start)


def extract_assertions(conceptnet_path, knowledge_store):
    # nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])
    nlp = spacy.load('en', parser=False, entity=False, matcher=False)
    rel2sf = dict()
    counter = 0
    with gzip.GzipFile(conceptnet_path) as f:
        for l in f:
            l = l.decode('utf-8')
            if not is_valid(l):
                continue
            if counter % 100000 == 0:
                logger.info('%d assertions added' % counter)
            try:
                split = l.strip().split("\t")
                [rel, subj, obj] = split[1:4]
                rel, subj, obj = only_string(rel), only_string(subj), only_string(obj)
                if subj == obj:
                    continue

                surface_form = rel2sf.get(rel)
                if surface_form is None:
                    j = json.loads(split[4])
                    surface_form = j.get("surfaceText", "[[%s]] %s [[%s]]" % (subj, rel, obj))
                    surface_form = surface_form.replace("[[%s]]" % subj, "[[subj]]").replace("[[%s]]" % obj, "[[obj]]")
                    rel2sf[rel] = surface_form

                subj_start = surface_form.find("[[subj]]")
                obj_start = surface_form.find("[[obj]]")
                if subj_start < 0 or obj_start < 0:
                    continue
                if subj_start < obj_start:
                    surface_form = surface_form.replace("[[subj]]", subj)
                    obj_start = surface_form.index("[[obj]]")
                    surface_form = surface_form.replace("[[obj]]", obj)
                else:
                    surface_form = surface_form.replace("[[obj]]", obj)
                    subj_start = surface_form.index("[[subj]]")
                    surface_form = surface_form.replace("[[subj]]", subj)

                tokens = nlp(surface_form)
                subj = lemmatized(tokens, subj_start, subj_start + len(subj))
                obj = lemmatized(tokens, obj_start, obj_start + len(obj))

                knowledge_store.add_assertion(surface_form, [subj], [obj], resource='conceptnet')
                counter += 1
            except Exception as e:
                logger.error('Error processing line: ' + l)
                raise e


if __name__ == '__main__':
    import json
    import logging
    import os
    import sys

    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logging.basicConfig(level=logging.INFO)

    store = KnowledgeStore(sys.argv[2], True)
    extract_assertions(sys.argv[1], store)
    store.save()
