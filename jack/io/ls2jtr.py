"""
This script converts data from the SemEval-2007 Task 10 on English Lexical
Substitution to the jtr format.
"""

import json
import xmltodict
import re
import os


def load_substitituons(path):
    subs = {}
    with open(path, "r") as f:
        for line in f.readlines()[1:]:
            splits = line.split(" :: ")
            id = splits[0].split(" ")[1]
            sub = [x[:-2] for x in splits[1].split(";")][:-1]
            print(id, sub)
            subs[id] = sub
    return subs


if __name__ == "__main__":
    CLOZE_STYLE = False

    for corpus_name in ["trial"]:
        file_path = "./jack/data/LS/%s/lexsub_%s_cleaned.xml" % (
        corpus_name, corpus_name)
        subs_path = "./jack/data/LS/%s/gold.%s" % (corpus_name, corpus_name)
        subs = load_substitituons(subs_path)

        with open(file_path) as fd:
            file_text = fd.read().replace("&", "&#038;")
            corpus = xmltodict.parse(file_text)["corpus"]

            jtr = {"meta": "SemEval-2007 Task 10: Lexical Substitution"}

            instances = []

            for lexelt in corpus["lexelt"]:
                for instance in lexelt["instance"]:
                    # fixme: not sure what is happening here
                    if str(instance) != "@id" and str(instance) != "context":
                        context = instance["context"]
                        id = instance["@id"]
                        word = re.search('_START_\w+_END_', context).group(0)[
                               7:-5]
                        context_masked = re.sub('_START_\w+_END_', 'XXXXX',
                                                context)
                        context_recovered = re.sub('_START_\w+_END_', word,
                                                   context)
                        context_tokenized = context.split(" ")
                        word_position = [i for i, word in enumerate(context_tokenized) if word.startswith('_START_')][0]

                        # print("%s\t%s\t%s" % (id, word, context_masked))

                        if CLOZE_STYLE:
                            queb = {'id': id, 'support': [], 'questions': [
                                {'question': context_masked,
                                 'answers': [
                                     {'text': word}
                                 ]}
                            ]}
                        else:
                            queb = {'id': id,
                                    'support': [{'text': context_recovered}],
                                    'questions': [
                                        {'question': str(word_position),
                                         'answers': [
                                             {'text': sub} for sub in subs[id]
                                         ]}
                                    ]}

                        instances.append(queb)

            jtr["instances"] = instances

            with open("./jack/data/LS/%s/lexsub_%s_cleaned.jsonl" % \
                              (corpus_name, corpus_name), 'w') as outfile:
                json.dump(jtr, outfile, indent=2)

            # create snippet
            jtr['instances'] = jtr['instances'][:10]


            def save_debug(directory_path, file_name):
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)

                with open(directory_path + "/" + file_name, 'w') as outfile:
                    json.dump(jtr, outfile, indent=2)


            save_debug("./data/LS/debug", "lexsub_debug_cleaned.jsonl")
            save_debug("./data/LS", "snippet.jack.json")
