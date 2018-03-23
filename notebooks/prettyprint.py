class QAPrettyPrint:
    def __init__(self, support, span):
        self.support = support
        self.span = span

    def _repr_html_(self):
        start, end = self.span
        pre_highlight = self.support[:start]
        highlight = self.support[start:end]
        post_highlight = self.support[end:]
        
        def _highlight(text):
            return '<span style="background-color: #ff00ff; color: white">' + text + '</span>'
        
        text = pre_highlight + _highlight(highlight) + post_highlight
        return text.replace('\n', '<br>')

def print_nli(premise, hypothesis, label):
	print('{}\t--({})-->\t{}'.format(premise, label, hypothesis))
