# -*- coding: utf-8 -*-

import json
import re
token_pattern = re.compile('[^ ]+')

import logging
logger = logging.getLogger(__name__)


"""Loads jtr JSON files and manages the transformation into components."""


def read_data(data_filename):
    """Reads jtr JSON and returns the dictionary."""
    with open(data_filename) as data_file:
        data = json.load(data_file)
        return data


def read_rich_data(data_filename):
    """Transforms a jtr JSON file into a list of RichInstances (S,Q(A))."""
    return [RichInstance(x) for x in read_data(data_filename)]


class RichInstance(object):
    """Holds lists of supports and questions and manages their iteration."""
    def __init__(self, instance):
        self.instance = instance
        self.supports = [Support(x) for x in self.instance['support']]
        self.questions = [Question(x) for x in self.instance['questions']]

    def vocab(self):
        """Gets the set of tokens from all questions and support objects."""
        tokens = set()
        for question in self.questions:
            for token in question.tokens:
                tokens.add(token)
        for support in self.supports:
            for token in support.tokens:
                tokens.add(token)
        return tokens

    def question_support_pairs(self):
        """Yields nested pairs of question and support objects."""
        for q in self.questions:
            for s in self.supports:
                yield q, s


class Question(object):
    """Class which holds answers and question as text."""
    def __init__(self, qdict):
        self.source = qdict
        self.text = qdict['question']
        self.answers = [Answer(x) for x in qdict['answers']]
        self.tokens = self.text.split(' ')


class Answer(object):
    """Class which holds the text or span of the answer."""
    def __init__(self, adict):
        self.adict = adict
        self.text = adict['text']
        self.span = adict['span']


class Support(object):
    """Holds a support which is a text or an interval between tokens (span)."""
    def __init__(self, sdict):
        self.sdict = sdict
        self.text = sdict['text']
        if 'tokens' in sdict:
            self.token_offsets = sdict['tokens']
        else:
            self.token_offsets = [m.span() for m in token_pattern.finditer(m)]
        self.tokens = [self.text[span[0]:span[1]] for span in self.token_offsets]

    def token_from_char(self, char_offset):
        """Returns string between zero and the char_offset."""
        for i,t in enumerate(self.token_offsets):
            if char_offset <= t[1]:
                return i
        return -1


def jtr_load(path, max_count=None, **options):
    """
    General-purpose loader for jtr files
    Makes use of user-defined options for supports, questions, candidates, answers and only read in those
    things needed for model, e.g. if the dataset contains support, but the user defines support_alts == 'none'
    because they want to train a model that does not make use of support, support information in dataset is not read in

    User options for jtr model/dataset attributes are:
    support_alts = {'none', 'single', 'multiple'}
    question_alts = answer_alts = {'single', 'multiple'}
    candidate_alts = {'open', 'per-instance', 'fixed'}
    """

    reading_dataset = json.load(path)

    def textOrDict(c):
        if isinstance(c, dict):
            c = c["text"]
        return c

    # The script reads into those lists. If IDs for questions, supports or targets are defined, those are ignored.
    questions = []
    supports = []
    answers = []
    candidates = []
    global_candidates = []
    count = 0
    if "globals" in reading_dataset:
        global_candidates = [textOrDict(c) for c in reading_dataset['globals']['candidates']]

    for instance in reading_dataset['instances']:
        question, support, answer, candidate = "", "", "", ""  # initialisation
        if max_count is None or count < max_count:
            if options["supports"] == "single":
                support = textOrDict(instance['support'][0])
            elif options["supports"].startswith("multiple"):
                support = [textOrDict(c) for c in instance['support']]
            if options["questions"] == "single":
                question = textOrDict(instance['questions'][0]["question"]) # if single, just take the first one, could also change this to random
                if options["answers"] == "single":
                    answer = textOrDict(instance['questions'][0]['answers'][0]) # if single, just take the first one, could also change this to random
                elif options["answers"] == "multiple":
                    answer = [textOrDict(c) for c in instance['questions'][0]['answers']]
                if options["candidates"] == "per-instance":
                    candidate = [textOrDict(c) for c in instance['questions'][0]['candidates']]

            elif options["questions"] == "multiple":
                answer = []
                candidate = []
                question = [textOrDict(c["question"]) for c in instance['questions']]
                if options["answers"] == "single":
                    answer = [textOrDict(c["answers"][0]) for c in instance['questions']]
                elif options["answers"] == "multiple":
                    answer = [textOrDict(c) for q in instance['questions'] for c in q["answers"]]
                if options["candidates"] == "per-instance":
                    candidate = [textOrDict(c) for quest in instance["questions"] for c in quest["candidates"]]

            if options["supports"] == "multiple_flat":
                for s in support:
                    supports.append(s)

                    if options["candidates"] == "fixed":
                        candidates.append(global_candidates)
                    if options["candidates"] != "fixed":
                        candidates.append(candidate)

                    questions.append(question)
                    answers.append(answer)

            else:
                if options["candidates"] == "fixed":
                    candidates.append(global_candidates)

                questions.append(question)
                answers.append(answer)
                if options["supports"] != "none":
                    supports.append(support)
                if options["candidates"] != "fixed":
                    candidates.append(candidate)

            count += 1

    logger.info("Loaded %d instances from %s" % (len(questions), path.name))
    if options["supports"] != "none":
        return {'question': questions, 'support': supports, 'answers': answers, 'candidates': candidates}
    else:
        return {'question': questions, 'answers': answers, 'candidates': candidates}
