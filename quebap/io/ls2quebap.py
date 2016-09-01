"""
This script converts data from the SemEval-2007 Task 10 on English Lexical
Substitution to the quebap format.
"""

import json
import sys
import xmltodict
import pprint
import re

if __name__ == "__main__":
    for corpus_name in ["trial"]:
        file_path = "./quebap/data/LS/%s/lexsub_%s_cleaned.xml" % (corpus_name, corpus_name)
        with open(file_path) as fd:
            file_text = fd.read().replace("&", "&#038;")
            corpus = xmltodict.parse(file_text)["corpus"]

            quebap = {"meta": "SemEval-2007 Task 10: Lexical Substitution"}

            instances = []

            for lexelt in corpus["lexelt"]:
                for instance in lexelt["instance"]:
                    # fixme: not sure what is happening here
                    if str(instance) != "@id" and str(instance) != "context":
                        context = instance["context"]
                        id = instance["@id"]
                        word = re.search('_START_\w+_END_', context).group(0)[7:-5]
                        context_masked = re.sub('_START_\w+_END_', 'XXXXX', context)

                        # print("%s\t%s\t%s" % (id, word, context_masked))

                        queb = {}
                        queb['id'] = id
                        queb['support'] = []
                        queb['questions'] = [
                            {'question': context_masked,
                             'answers': [
                                 {'text': word}
                             ]}
                        ]

                        instances.append(queb)

            quebap["instances"] = instances

            with open("./quebap/data/LS/%s/lexsub_%s_cleaned.jsonl" % \
                              (corpus_name, corpus_name), 'w') as outfile:
                json.dump(quebap, outfile, indent=2)
