"""
Here we define light data structures to store the input to jtr readers, and their output.
"""

from typing import NamedTuple, List, Tuple
from jtr.load.read_jtr import jtr_load

Answer = NamedTuple("Answer", [('text', str), ('score', float)])
Input = NamedTuple("Input", [('support', List[str]), ('question', str), ('candidates', List[str])])


def load_labelled_data(path, max_count=None, **options) -> List[Tuple[Input, Answer]]:
    """
    This function loads a jtr json file with labelled answers from a specific location.
    Args:
        path: the location to load from.
        max_count: how many instances to load at most
        **options: options to pass on to the loader. TODO: what are the options

    Returns:
        A list of input-answer pairs.

    """
    dict_data = jtr_load(path, max_count, **options)
    if "support" not in dict_data:
        dict_data["support"] = []

    def to_list(text_or_list):
        if isinstance(text_or_list, str):
            return [text_or_list]
        else:
            return text_or_list

    def convert_instance(index):
        support = to_list(dict_data['support'][index])
        question = dict_data['question'][index]
        candidates = dict_data['candidates'][index]
        answer = dict_data['answer'][index]
        return Input(support, question, candidates), Answer(answer, 1.0)

    result = [convert_instance(i) for i in range(0, len(dict_data['question']))]
    return result
