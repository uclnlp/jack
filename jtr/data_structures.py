"""
Here we define light data structures to store the input to jtr readers, and their output.
"""

from typing import NamedTuple, List

Answer = NamedTuple("Answer", [('text', str), ('score', float)])
Input = NamedTuple("Input", [('support', List[str]), ('question', str), ('candidates', List[str])])
