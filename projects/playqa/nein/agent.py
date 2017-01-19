import nose.tools
import abc

READ = "READ"
QUIT = "QUIT"
OK = "OK"
HELP = "HELP"
ACK = "ACK"
RESPONSE = "RESPONSE"
INIT = "INIT"


class Provenance(object):
    """
    Class to store information about where the agent gets her information from.
    """

    def __init__(self, sentences, attention):
        self.sentences = sentences
        self.attention = attention

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(vars(self)) + ')'


class Output(object):
    """
    Representing the output of an agent.
    """

    def __init__(self, response, status=OK,
                 scores=None, candidates=None, provenance: Provenance = None,
                 help_msg=[]):
        self.response = response
        self.scores = scores
        self.candidates = candidates
        self.status = status
        self.provenance = provenance
        self.help_msg = help_msg

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(vars(self)) + ')'


class Agent(metaclass=abc.ABCMeta):
    """
    An agent takes a string input and returns Output objects.
    """

    @abc.abstractmethod
    def __call__(self, input: str):
        """

        Args:
            input: a string

        Returns: Output

        """
        pass


def input_reader_func():
    return input('> ')


def simple_loop(agent,
                input_reader=None,
                output_printer=lambda output: print(output.response)):
    """
    Simple loop for interacting with an agent through the console.
    :param agent: a function that receives a string inputs and returns an Output instance.
    :param input_reader: a procedure that returns input strings, typically from the console.
    :param output_printer: a procedure that prints the output of the agent.
    """
    if input_reader is None:
       input_reader = input_reader_func

    status = None
    input = input_reader()
    while status != QUIT or input is None:
        response = agent(input)
        status = response.status
        output_printer(response)
        input = input_reader()


def test_simple_loop():
    """
    Test the loop by using a copying/remembering agent.
    """
    answers = []
    time_step = [0]
    inputs = ['Yo', 'What', QUIT]

    def agent(input):
        status = QUIT if input == QUIT else OK
        return Output(input, status)

    def input_reader():
        time_step[0] += 1
        return inputs[time_step[0] - 1]

    def output_printer(output):
        answers.append(output.response)

    simple_loop(agent, input_reader, output_printer)
    nose.tools.assert_equal(answers, inputs)

