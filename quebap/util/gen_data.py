from pycorenlp import StanfordCoreNLP
from enum import Enum
from collections import defaultdict
from nltk.tree import Tree
import copy
import abc


def tree_string(tree):
    return tree if isinstance(tree, str) else tree.label()


def find_labels(t, labels=('VP', 'CC', 'VP')):
    return find_tree(t, lambda t: tuple([tree_string(c) for c in t]) == labels)


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


examples = [
    "LOCATION1 is the trendiest place in the capital and completely shed the old image Thinking of moving to London",
    "LOCATION1 is quite a long way out of London , but its very green",
    "i live in the LOCATION1   wouldn't recommend it",
]

nlp = StanfordCoreNLP('http://localhost:9000')


class Consistency(Enum):
    yes = 1
    no = 2
    unknown = 3


def default_question():
    return "g"


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
    def __init__(self, support, question, answer,
                 support_trees=None,
                 question_trees=None,
                 answer_trees=None):
        self.answer = answer
        self.question = question
        self.support = support
        self.support_trees = support_trees if support_trees is not None else parse_trees(support)
        self.question_trees = question_trees if question_trees is not None else parse_trees(question)
        self.answer_trees = answer_trees if answer_trees is not None else parse_trees(answer)

    def copy(self, support=None, question=None, answer=None,
             support_trees=None, question_trees=None, answer_trees=None):
        result = copy.copy(self)
        if support is not None:
            result.support = support
        if support_trees is not None:
            result.support_trees = support_trees
        if answer is not None:
            result.answer = answer
        if answer_trees is not None:
            result.answer_trees = answer_trees
        return result

    def __str__(self):
        return ("Support:  {support}\n"
                "Question: {question}\n"
                "Answer:   {answer}").format(support=self.support, question=self.question, answer=self.answer)


class Action(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def do_action(self, grammar, proposal_queue):
        pass


class ProposeNextActions(Action):
    def __init__(self, instance):
        self.instance = instance

    def do_action(self, grammar: defaultdict, proposal_queue: list):
        tree = self.instance.support_trees[0]
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


class FixInstance(Action):
    def __init__(self, instance):
        self.instance = instance

    def do_action(self, grammar, proposal_queue: list):
        print(self.instance)
        answer = input("Is this correct (y/n)? ")
        if answer == '' or answer[0].lower() == 'n':
            to_fix = input("What do you want to fix ([s]upport/[q]uestion/[a]nswer)? ")
            if to_fix == '' or to_fix[0].lower() == 'a':
                fix = input("Answer? ")
                new_instance = self.instance.copy(answer=fix, answer_trees=parse_trees(fix))
                proposal_queue.append(ProposeNextActions(new_instance))


def incomplete_tree_to_string(tree):
    text = [l if isinstance(l, str) else "[ " + l.label() + " ]" for l in tree.leaves()]
    return " ".join(text)


class FixQuestionAnswer(Action):
    def __init__(self, instance):
        self.instance = instance

    def do_action(self, grammar, proposal_queue: list):
        print(self.instance)
        input_text = input("> Correct '[Question]? [Answer]' pair, empty if already connect: ")
        if input_text != '':
            question, answer = input_text.split("?")
            new_instance = self.instance.copy(answer=answer, answer_trees=parse_trees(answer))
        else:
            new_instance = self.instance
        proposal_queue.append(ProposeNextActions(new_instance))
        grammar['T'].append(new_instance)


class DropConjunct(Action):
    def do_action(self, grammar, proposal_queue):
        tree = self.instance.support_trees[0]
        rhs = transform_tree(tree, lambda t: t[self.conjunct_index_to_keep] if t == self.conjunct_parent else None)
        rhs_text = " ".join(rhs.leaves())
        new_instance = self.instance.copy(support=rhs_text, support_trees=[rhs])
        print(new_instance)
        current_input = input("> Is this still correct ([y]es/[n]o)? ")
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

            second_instance = self.instance.copy(support=rhs_with_nonterminal_text,
                                                 support_trees=[rhs_with_nonterminal])

            grammar['T'].append(new_instance)
            grammar['T'].append(second_instance)

            proposal_queue += [ProposeNextActions(new_instance)]

    def __init__(self, instance: Instance, conjunct_parent: Tree, conjunct_index_to_keep: int):
        self.instance = instance
        self.conjunct_index_to_keep = conjunct_index_to_keep
        self.conjunct_index_to_remove = 0 if self.conjunct_index_to_keep == 2 else 2
        self.conjunct_parent = conjunct_parent


class DropPP(Action):
    def do_action(self, grammar, proposal_queue):
        tree = self.instance.support_trees[0]
        rhs = transform_tree(tree, lambda t: t[0] if t == self.parent else None)
        rhs_text = " ".join(rhs.leaves())
        new_instance = self.instance.copy(support=rhs_text, support_trees=[rhs])
        print(new_instance)
        current_input = input("> Is this still correct ([y]es/[n]o)? ")


    def __init__(self, instance, parent):
        self.instance = instance
        self.parent = parent


class ChooseNextInstance(Action):
    def do_action(self, grammar, proposal_queue):
        proposal_queue.append(FixQuestionAnswer(Instance(examples[0], default_question(), default_answer())))
        del examples[0]

    def __init__(self, examples):
        self.examples = examples


def interaction_loop():
    queue = [ChooseNextInstance(examples)]
    grammar = defaultdict(list)

    while len(queue) > 0:
        action = queue.pop()
        action.do_action(grammar, queue)

    for non_terminal, rhs_list in grammar.items():
        print(non_terminal)
        for rhs in rhs_list:
            print(rhs)


interaction_loop()
