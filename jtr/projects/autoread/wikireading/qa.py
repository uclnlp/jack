

class QASetting:

    def __init__(self, question, answers, context):
        """
        :param question: list of indices
        :param answers:  list of list of indices
        :param context: list of indices
        :return:
        """
        self.question = question
        self.answers = answers
        self.context = context