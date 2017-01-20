# -*- coding: utf-8 -*-

import json

from enum import Enum
from collections import defaultdict

import copy
import abc
from random import shuffle
from os import path
import time
import sys
import traceback

import numpy as np
from pycorenlp import StanfordCoreNLP
from nltk.tree import Tree


adjective_dictionary = {}
replacement_dictionary = {}


def tree_string(tree):
    return tree if isinstance(tree, str) else tree.label()


def find_labels(t, labels=('VP', 'CC', 'VP')):
    return find_tree(t, lambda t: tuple([tree_string(c) for c in t]) == labels)


def find_trees_with_label(t, label="JJ"):
    trees = []
    if isinstance(t, str):
        return trees
    if not isinstance(t, str) and t.label() == label:
        return trees + [t]
    else:
        for child in t:
            trees += find_trees_with_label(child, label)
        return trees


def find_tree(t, predicate):
    if predicate(t):
        return [t]
    else:
        return [result
                for c in t if not isinstance(c, str)
                for result in find_tree(c, predicate)]


def transform_tree(tree, func, include_terminals=False):
    if not include_terminals and isinstance(tree, str):
        return tree
    result = func(tree)
    if result is None:
        if isinstance(tree, str):
            return tree
        else:
            children = tuple([transform_tree(c, func) for c in tree])
            return Tree(tree.label(), children)
    else:
        return result


def read_data(files):
    sentences = []
    for file in files:
        if path.isfile(file):
            with open(file, 'r') as myfile:
                json_txt = myfile.read()
                json_dicts = json.loads(json_txt)
                for json_dict in json_dicts:
                    sentence = json_dict['text']
                    if sentence.count("LOCATION1") <= 1 and sentence.count("LOCATION2") <= 1:
                        if "location2" not in sentence.lower():
                            sentences.append(sentence.replace("LOCATION1", "target_loc"))
                        else:
                            sent1 = sentence.replace("LOCATION1", "target_loc")
                            sent2 = sentence.replace("LOCATION2", "target_loc")
                            sentences.append(sent1)
                            sentences.append(sent2)
        else:
            print("file " + file + " Not Found!!")
    return sentences


def read_examples(mod="examples_left"):
    if mod == "raw":
        examples_all = read_data([data_dir + "single_train.json", data_dir + "multi_train.json"])
    elif mod == "mock":
        examples_all = [
            # "LOCATION1 is n't the best place to live , I use to work there LOCATION1 is not a very good place",
            # "LOCATION1 is the trendiest place in the capital and completely shed the old image Thinking of moving to London",
            # "target_loc is a pleasent , up-market , vibrant , cosmopolitan , young professional area",
            # "target_loc is an up-market, vibrant area",
            # "LOCATION2 in target_loc (London borough of richmond) is pretty good"
            "target_loc in Kent is a lovely ancient market town with a good fast rail service to LOCATION2"
            # "If you were to go out to somewhere like target_loc, then you would have a very long commute to the London",
            # "target_loc is pretty risky",
            # "I 'd say target_loc is pretty risky , i live not too far from there",
            # "The very centre of London -LRB- places like target_loc -RRB- are very well off and there are lower levels of crime",
            # "LOCATION1 is quite a long way out of London , but its very green",
            # "i live in the LOCATION1   wouldn't recommend it",
        ]
    elif mod == "examples_left":
        if path.exists(data_dir + "examples_left.json"):
            with open(data_dir + "examples_left.json") as data_file:
                data = json.load(data_file)
                examples_all = [instance['text'] for instance in data]
        else:
            examples_all = read_data([data_dir + "single_train.json", data_dir + "multi_train.json"])
    shuffle(examples_all)
    return examples_all


def check_for_special_commands(answer):
    if answer == 'q':
        raise ValueError("quit")
    elif answer == 'qw':
        raise ValueError("quit_save")
    elif answer == 'k':
        raise ValueError("skip")
    elif answer == 'sk':
        raise ValueError("skip_store")


class Consistency(Enum):
    yes = 1
    no = 2
    unknown = 3


def default_question():
    return "p"


def default_answer():
    return "~"


def parse_trees(text):
    parsed = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,parse',
        'outputFormat': 'json'
    })
    trees = [Tree.fromstring(parsed['sentences'][i]['parse']).freeze()
             for i in range(0, len(parsed['sentences']))]
    return tuple(trees)


class Instance:
    def __init__(self, id, support, question, answer,
                 support_trees=None,
                 question_trees=None,
                 answer_trees=None):
        self.id = id
        self.answer = answer
        self.question = question
        self.support = support
        self.support_trees = support_trees if support_trees is not None else parse_trees(support)
        self.question_trees = question_trees if question_trees is not None else parse_trees(question)
        self.answer_trees = answer_trees if answer_trees is not None else parse_trees(answer)

    def copy(self, id=None, support=None, question=None, answer=None,
             support_trees=None, question_trees=None, answer_trees=None):
        result = copy.copy(self)
        if id is not None:
            result.id = id
        if support is not None:
            result.support = support
        if support_trees is not None:
            result.support_trees = support_trees
        if answer is not None:
            result.answer = answer
        if question is not None:
            result.question = question
        if answer_trees is not None:
            result.answer_trees = answer_trees
        return result

    def __str__(self):
        return ("\nID: {id}\n"
                "Support:  {support}\n"
                "Question: {question}\n"
                "Answer:   {answer}").format(id=self.id, support=self.support, question=self.question,
                                             answer=self.answer)


class Log():
    def __init__(self, instance, action, input, time):
        self.instance = instance
        self.action = action
        self.input = input
        self.time = time

    def __str__(self):
        return ("\nInstance:  {instance}\n"
                "Action: {action}\n"
                "Time: {time}").format(instance=self.instance, action=self.action, time=self.time)


class Action(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def do_action(self, grammar, proposal_queue):
        pass


class ProposeNextActions(Action):
    def __init__(self, instance):
        self.instance = instance

    def do_action(self, grammar: defaultdict, proposal_queue: list):
        tree = self.instance.support_trees[0]

        low_level_S_trees = [result
                             for root_child in tree
                             for root_grand_parent in root_child
                             for result in
                             find_tree(root_grand_parent,
                                       lambda t: len(t) > 1 and not isinstance(t, str) and t.label() == 'S')]  # ???

        if len(low_level_S_trees) > 0:
            for s_tree in low_level_S_trees:
                proposal_queue += [KeepOnlyTree(self.instance, s_tree)]
        # conjuncts = find_labels(tree, labels=('VP', 'CC', 'VP'))
        conjuncts = find_tree(tree, lambda t: len(t) == 3 and tree_string(t[0]) == tree_string(t[2]) and tree_string(
            t[1]) == 'CC')
        if len(conjuncts) > 0:
            for vp_cc_vp in conjuncts:
                proposal_queue += [DropConjunct(self.instance, vp_cc_vp, 2), DropConjunct(self.instance, vp_cc_vp, 0)]
            return

        pp_attachments = find_tree(tree, lambda t: len(t) == 2 and tree_string(t[1]) == 'PP')
        if len(pp_attachments) > 0:
            for pp_attachment in pp_attachments:
                proposal_queue += [DropPP(self.instance, pp_attachment)]
            return

        frag_labels = {'SBAR', 'FRAG'}
        with_frags = find_tree(tree, lambda t: len(t) >= 2 and tree_string(t[-2]) == ',' and tree_string(
            t[-1]) in frag_labels)
        if len(with_frags) > 0:
            for with_frag in with_frags:
                proposal_queue += [DropFragmentOrSBar(self.instance, with_frag)]
            return


class FixInstance(Action):
    def __init__(self, instance):
        self.instance = instance

    def do_action(self, grammar, proposal_queue: list):
        print(self.instance)
        answer = input("Is this correct (y/n)? ")
        if answer == '' or answer[0].lower() == 'n':
            to_fix = input("What do you want to fix ([s]upport/[q]uestion/[a]nswer)? ")
            check_for_special_commands(to_fix)
            if to_fix == '' or to_fix[0].lower() == 'a':
                fix = input("Answer? ")
                new_instance = self.instance.copy(id=time.time(), answer=fix, answer_trees=parse_trees(fix))
                if "target_loc" in new_instance.support:
                    proposal_queue.append(ProposeNextActions(new_instance))


def incomplete_tree_to_string(tree):
    text = [l if isinstance(l, str) else "[ " + l.label() + " ]" for l in tree.leaves()]
    return " ".join(text)


class FixQuestionAnswer(Action):
    def __init__(self, instance):
        self.instance = instance

    def do_action(self, grammar, proposal_queue: list):
        print(self.instance)
        start = time.time()
        input_text = input("> Correct '[Question]? [Answer]' pair, empty if already correct: ")
        elapsed = time.time() - start
        check_for_special_commands(input_text)

        if input_text != '':
            question, answer = input_text.split("?")
            new_instance = self.instance.copy(id=time.time(), question=question, answer=answer,
                                              answer_trees=parse_trees(answer))
        else:
            new_instance = self.instance
        proposal_queue.append(ProposeNextActions(new_instance))
        grammar.append(new_instance)
        action_name = self.__class__.__name__
        logs.append(Log(new_instance, action_name, input_text, elapsed))


class AskForAdditionalAnnotation(Action):
    def __init__(self, instance):
        self.instance = instance

    def do_action(self, grammar, proposal_queue: list):
        self.instance = self.instance.copy(id=time.time(), answer="~", question="~",
                                           answer_trees=self.instance.answer_trees)
        print(self.instance)
        start = time.time()
        input_text = input("> Additional '[Question]? [Answer]' pair, empty to proceed: ")
        check_for_special_commands(input_text)
        elapsed = time.time() - start
        action_name = self.__class__.__name__

        if input_text != '':
            question, answer = input_text.split("?")
            new_instance = self.instance.copy(id=time.time(), answer=answer, question=question,
                                              answer_trees=parse_trees(answer))
            logs.append(Log(new_instance, action_name, input_text, elapsed))
            proposal_queue.append(AskForAdditionalAnnotation(new_instance))
            proposal_queue.append(ProposeNextActions(new_instance))
            grammar.append(new_instance)


class DropConjunct(Action):
    def do_action(self, grammar, proposal_queue):
        tree = self.instance.support_trees[0]
        rhs = transform_tree(tree, lambda t: t[self.conjunct_index_to_keep] if t == self.conjunct_parent else None)
        rhs_text = " ".join(rhs.leaves())
        new_instance = self.instance.copy(id=time.time(), support=rhs_text, support_trees=[rhs])
        if "target_loc" in new_instance.support:
            print(new_instance)
            start = time.time()
            current_input = input("> Is this still correct ([y]es/[n]o)? ")
            check_for_special_commands(current_input)
            elapsed = time.time() - start
            action_name = self.__class__.__name__
            logs.append(Log(new_instance, action_name, current_input, elapsed))
            if current_input == '' or current_input[0] == 'y':
                # todo: create one version where a generic VP is used instead of the VP to drop
                new_conjunct_children = [child for child in self.conjunct_parent]
                replacement_label = self.conjunct_parent[self.conjunct_index_to_remove].label()
                new_conjunct_children[self.conjunct_index_to_remove] = Tree(replacement_label, [
                    "[ " + replacement_label + " ]"])

                new_conjunct_parent = Tree(self.conjunct_parent.label(), new_conjunct_children)

                rhs_with_nonterminal = \
                    transform_tree(tree, lambda t: new_conjunct_parent if t == self.conjunct_parent else None)

                rhs_with_nonterminal_text = incomplete_tree_to_string(rhs_with_nonterminal)

                second_instance = self.instance.copy(id=time.time(), support=rhs_with_nonterminal_text,
                                                     support_trees=[rhs_with_nonterminal])

                grammar.append(new_instance)
                grammar.append(second_instance)
                action = ProposeNextActions(new_instance)
                proposal_queue += [action]

    def __init__(self, instance: Instance, conjunct_parent: Tree, conjunct_index_to_keep: int):
        self.instance = instance
        self.conjunct_index_to_keep = conjunct_index_to_keep
        self.conjunct_index_to_remove = 0 if self.conjunct_index_to_keep == 2 else 2
        self.conjunct_parent = conjunct_parent


def ask_user(question, choices=('yes', 'no')):
    choice_strings = ["[{abbr}]{rest}".format(abbr=choice[0], rest=choice[1:]) for choice in choices]
    answer = input("> {question} ({choices})".format(question=question, choices="/".join(choice_strings))).strip()
    check_for_special_commands(answer)
    if answer == "":
        return choices[0]
    for choice in choices:
        if answer.lower() == choice[0].lower() or answer.lower() == choice.lower():
            return choice
    raise RuntimeError("Wrong input {}".format(answer))


class DropPP(Action):
    def do_action(self, grammar, proposal_queue):
        tree = self.instance.support_trees[0]
        rhs = transform_tree(tree, lambda t: t[0] if t == self.parent else None)
        rhs_text = " ".join(rhs.leaves())
        new_instance = self.instance.copy(id=time.time(), support=rhs_text, support_trees=[rhs])
        if "target_loc" in new_instance.support:
            print(new_instance)
            start = time.time()
            answer = ask_user("Is this still correct", ("yes", "no"))
            check_for_special_commands(answer)
            elapsed = time.time() - start
            action_name = self.__class__.__name__
            logs.append(Log(new_instance, action_name, answer, elapsed))
            if answer == "yes":
                grammar.append(new_instance)
                proposal_queue += [ProposeNextActions(new_instance)]
                generic_parent = Tree(self.parent.label(), [self.parent[0], Tree("PP", ["[ PP ]"])])
                rhs_2 = transform_tree(tree, lambda t: generic_parent if t == self.parent else None)
                rhs_2_text = " ".join(rhs_2.leaves())
                new_instance_2 = self.instance.copy(id=time.time(), support=rhs_2_text, support_trees=[rhs_2])
                print(new_instance_2)
                # pp_proposed_replacements = []
                # if "PP" in replacement_dictionary:
                #     pp_proposed_replacements = replacement_dictionary["PP"]
                # start = time.time()
                # answer = input(
                #     "> Do you want to replace [ PP ] with one or more propositions (comma separated) to keep the same sentiment? (no to dismiss template) enter to use the template [ PP ].")
                # check_for_special_commands(answer)
                # if answer == "":
                #     grammar.append(new_instance_2)
                #     logs.append(Log(new_instance_2, action_name, answer, time.time() - start))
                # elif answer != "n" and answer != "no":
                #     pp_replacements = answer.split(",")
                #     if len(pp_replacements) > 0:
                #         elapsed = (time.time() - start) / len(pp_replacements)
                #         for pp_replacement in pp_replacements:
                #             support_pp = new_instance_2.support.replace("[ PP ]", pp_replacement)
                #             new_instance_pp = Instance(time.time(), support=support_pp,
                #                                        question=new_instance_2.question,
                #                                        answer=new_instance_2.answer, action=self.__class__.__name__)
                #             grammar.append(new_instance_pp)
                #             logs.append(Log(new_instance_pp, action_name, answer, elapsed))

    def __init__(self, instance, parent):
        self.instance = instance
        self.parent = parent


class ReplaceAdjective(Action):
    def do_action(self, grammar, proposal_queue):
        tree = self.instance.support_trees[0]
        adj_trees = find_trees_with_label(tree, label='JJ')
        for adj_tree in adj_trees:
            adjective = adj_tree.leaves()[0]
            start = time.time()

            proposed_replacements = []
            if adjective in adjective_dictionary:
                proposed_replacements = adjective_dictionary[adjective]
                answer = input(
                    "\n> Can you to replace \"" + adjective + "\" with any of " + str(
                        proposed_replacements) + " without changing the sentiment? (n). Provide a comma separate list to provide other replacements. Empty to proceed: ")
            else:
                answer = input(
                    "\n> Can you replace \"" + adjective + "\" with other adjectives without changing the sentiment? Provide a comma separate list to provide replacements. Empty to proceed: ")

            check_for_special_commands(answer)
            if answer == "" or answer == "y":
                replacements = proposed_replacements
            elif answer == "n":
                continue
            else:
                replacements = [s for s in answer.strip().split(",") if s != ""]

            if len(replacements) > 0:
                elapsed = (time.time() - start) / (len(replacements) * len(grammar))
                new_instances = []
                for replacement in replacements:
                    for instance in grammar:
                        support_text = instance.support.replace(adjective, replacement)
                        new_instance = instance.copy(id=time.time(), support=support_text, question=instance.question,
                                                     answer=instance.answer)
                        new_instances.append(new_instance)
                        logs.append(Log(new_instance, self.__class__.__name__, replacement, elapsed))
                grammar += new_instances
                adjective_dictionary[adjective] = np.unique(np.array(list(proposed_replacements) + list(replacements)))

    def __init__(self, instance):
        self.instance = instance


class CutSlice(Action):
    def do_action(self, grammar, proposal_queue):
        start = time.time()
        answer = input(
            "\n> Do you want to annotate specific part of the sentence? empty to proceed:  ")
        elapsed = time.time() - start
        check_for_special_commands(answer)
        if answer != "":
            exs = answer.split("|")
            for ex in exs:
                examples.insert(0, ex)
        logs.append(Log(self.instance, self.__class__.__name__, answer, time=elapsed))

    def __init__(self, instance):
        self.instance = instance


class KeepOnlyTree(Action):
    def do_action(self, grammar, proposal_queue):
        rhs = self.tree
        rhs_text = " ".join(rhs.leaves())
        new_instance = self.instance.copy(id=time.time(), support=rhs_text, support_trees=[rhs])
        if "target_loc" in new_instance.support:
            print(new_instance)
            start = time.time()
            answer = ask_user("Is this still correct", ("yes", "no"))
            check_for_special_commands(answer)
            elapsed = time.time() - start
            action_name = self.__class__.__name__
            logs.append(Log(new_instance, action_name, answer, elapsed))
            if answer == "yes":
                grammar.append(new_instance)
                proposal_queue += [ProposeNextActions(new_instance)]

    def __init__(self, instance, tree):
        self.tree = tree
        self.instance = instance


class DropFragmentOrSBar(Action):
    def do_action(self, grammar, proposal_queue):
        tree = self.instance.support_trees[0]
        rhs = transform_tree(tree, lambda t: Tree(t.label(), t[:-2]) if t == self.parent else None)
        rhs_text = " ".join(rhs.leaves())
        new_instance = self.instance.copy(id=time.time(), support=rhs_text, support_trees=[rhs])
        if "target_loc" in new_instance.support:
            print()
            print(new_instance)
            start = time.time()
            answer = ask_user("Is this still correct", ("yes", "no"))
            check_for_special_commands(answer)
            elapsed = time.time() - start
            action_name = self.__class__.__name__
            logs.append(Log(new_instance, action_name, answer, time=elapsed))
            if answer == "yes":
                generic_parent = Tree(self.parent.label(), self.parent[:-1] + [Tree("SBAR", ["[ SBAR ]"])])
                rhs_2 = transform_tree(tree, lambda t: generic_parent if t == self.parent else None)
                rhs_2_text = " ".join(rhs_2.leaves())
                new_instance_2 = self.instance.copy(id=time.time(), support=rhs_2_text, support_trees=[rhs_2])
                grammar.append(new_instance)
                grammar.append(new_instance_2)
                logs.append(Log(new_instance_2, action_name, answer, time=0))
                proposal_queue += [ProposeNextActions(new_instance)]

    def __init__(self, instance, parent):
        self.instance = instance
        self.parent = parent


class ChooseNextInstance(Action):
    def do_action(self, grammar, proposal_queue):
        proposal_queue.append(CutSlice(Instance(time.time(), self.examples[0], default_question(), default_answer())))
        proposal_queue.append(
            ReplaceAdjective(Instance(time.time(), self.examples[0], default_question(), default_answer())))
        proposal_queue.append(
            AskForAdditionalAnnotation(Instance(time.time(), self.examples[0], default_question(), default_answer())))
        proposal_queue.append(
            FixQuestionAnswer(Instance(time.time(), self.examples[0], default_question(), default_answer())))
        del self.examples[0]

    def __init__(self, examples):
        self.examples = examples


def write_annotations(grammar, start_time, end_time):
    time_elapsed = int(end_time - start_time)
    output_dict = []
    for non_terminal, rhs_list in grammar.items():
        print(non_terminal)
        for rhs in rhs_list:
            rhs_dict = {}
            rhs_dict['support'] = rhs.support
            rhs_dict['question'] = rhs.question
            rhs_dict['answer'] = rhs.answer
            rhs_dict['id'] = rhs.id
            print(rhs)
            output_dict.append(rhs_dict)

    # Write to jason and file
    json_ser = json.dumps(output_dict)
    out_file = open(
        out_dir + "annotation_" + str(int(end_time)) + "_elapsed_" + str(time_elapsed) + "_.json",
        'w')
    out_file.write(json_ser)
    out_file.close()


def write_examples_left(examples_left):
    dicts = [{"text": e} for e in examples_left]
    json_ser_ex = json.dumps(dicts)
    out_file = open(data_dir + "examples_left.json", 'w')
    out_file.write(json_ser_ex)
    out_file.close()


def write_logs(end_time):
    log_dicts = []
    for log in logs:
        log_dict = {}
        log_dict['support'] = log.instance.support
        log_dict['question'] = log.instance.question
        log_dict['answer'] = log.instance.answer
        log_dict['action'] = log.action
        log_dict['input'] = log.input
        log_dict['time'] = log.time
        log_dict['id'] = log.instance.id
        log_dicts.append(log_dict)

    # Write to jason and file
    json_ser = json.dumps(log_dicts)
    out_file = open(
        out_dir + "logs_" + str(int(end_time)) + "_.json",
        'w')
    out_file.write(json_ser)
    out_file.close()


def interaction_loop(save=True):
    queue = [ChooseNextInstance(examples)]

    grammar = defaultdict(list)
    start_time = time.time()
    instance_grammar = []
    try:
        while len(queue) > 0:
            try:

                action = queue.pop()
                action.do_action(instance_grammar, queue)

                if len(queue) == 0:
                    answer = ask_user("\n>>> Do you want to add instances of the above sentence to grammar?",
                                      ("yes", "no"))
                    if answer == '' or answer == 'yes':
                        grammar['T'] += instance_grammar
                        instance_grammar = []
                    if len(examples) > 0:
                        print("-" * 50)
                        instance = ChooseNextInstance(examples)
                        queue += [instance]
            except Exception as ex:
                if type(ex).__name__ == 'ValueError' and ('quit' == str(ex) or 'quit_save' == str(ex)):
                    grammar['T'] += instance_grammar
                    raise ex
                if type(ex).__name__ == 'ValueError' and 'skip_store' == str(ex):
                    grammar['T'] += instance_grammar
                    instance_grammar = []
                    while len(queue) > 0:
                        queue.pop()
                    if len(examples) > 0:
                        queue += [ChooseNextInstance(examples)]
                    print("-" * 50)
                elif type(ex).__name__ == 'ValueError' and 'skip' == str(ex):
                    instance_grammar = []
                    while len(queue) > 0:
                        queue.pop()
                    if len(examples) > 0:
                        queue += [ChooseNextInstance(examples)]
                    print("-" * 50)
                else:
                    # raise ex
                    traceback.print_exc(file=sys.stdout)
                    print(repr(ex))
    except ValueError as err:
        err_msg = str(err)
        end_time = time.time()
        if 'quit_save' == err_msg:
            print("Quitting and writing ..")
        if 'quit' == err_msg:
            answer = ask_user("Do you want to save before quitting?", ("yes", "no"))
            if answer == "no":
                save = False
                print("Quitting and writing ..")
    if save:
        write_annotations(grammar, start_time, end_time)
        write_examples_left(examples)
        write_logs(end_time)

    for instance in grammar['T']:
        print(instance)


logs = []
nlp = StanfordCoreNLP('http://localhost:9000')

data_dir = "../data/urban/raw/"
out_dir = "../data/urban/annotated/"

# examples = read_examples("mock")
examples = read_examples("examples_left")
interaction_loop(save=True)
