class Expr:
    def __repr__(self):
        return self.__dict__.__repr__()


class Variable(Expr):
    def __init__(self, name):
        self.name = name


class Constant(Expr):
    def __init__(self, name):
        self.name = name


class Predicate(Expr):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, self.__class__) and other.name == self.name

    def __hash__(self):
        return self.name.__hash__()


class Atom(Expr):
    def __init__(self, predicate: Predicate, *arguments):
        self.predicate = predicate
        self.arguments = arguments


class Clause(Expr):
    def __init__(self, head, *body):
        self.head = head
        self.body = body


from parsimonious.grammar import Grammar, NodeVisitor

grammar = Grammar(
    """
    clause     = atom ( ":-" _ atom_list)?
    atom_list  = atom  ("," _ atom_list)? _
    atom       = predicate "(" _ term_list ")" _
    term_list  = term  ("," _ term_list)? _
    term       = constant / variable
    predicate  = low_id / string
    constant   = low_id / string
    variable   = up_id
    low_id     = ~"[a-z][a-z A-Z 0-9]*"
    up_id      = ~"[A-Z][a-z A-Z 0-9]*"
    string     = ~r"'[^']*'" / ~r"\\"[^\\"]*\\""
    _          = skip*
    skip       = ~r"\s+"
    """)


class ClauseVisitor(NodeVisitor):
    def visit_clause(self, node, visited_children):
        print("Clause")
        if len(visited_children[1]) == 0:
            return None
        else:
            print(visited_children)
            head, ((_, _, body),) = visited_children
            return Clause(head, body)

    def visit_predicate(self, _, visited_children):
        return Predicate(visited_children[0])

    def visit_constant(self, _, visited_children):
        return Constant(visited_children[0])

    def visit_variable(self, _, visited_children):
        return Variable(visited_children[0])

    def visit_term(self, _, visited_children):
        print("In Term")
        print(visited_children)
        return visited_children[0]

    def visit_term_list(self, _, visited_children):
        print("TermList")
        print(visited_children)
        if len(visited_children[1]) == 0:
            return visited_children[:1]
        else:
            head, ((_, _, tail),), _ = visited_children
            return [head] + tail

    def visit_atom_list(self, _, visited_children):
        print("AtomList")
        print(visited_children)
        if len(visited_children[1]) == 0:
            return visited_children[:1]
        else:
            print(visited_children)
            head, ((_, _, tail),), _ = visited_children
            return [head] + tail

    def visit__(self, _, visited_children):
        print("____")
        return []

    def visit_atom(self, _, visited_children):
        print("In Atom")
        print(visited_children)
        return Atom(visited_children[0], *visited_children[3])

    def visit_low_id(self, node, _):
        return node.full_text[node.start:node.end]

    def visit_up_id(self, node, _):
        return node.full_text[node.start:node.end]

    def visit_string(self, node, _):
        return node.full_text[node.start:node.end]

    def generic_visit(self, node, visited_children):
        return visited_children


parsed = grammar.parse("""sister(X,Y) :- brother(Y,"X asd "), what(Z,X)""")


def parse_clause(text):
    grammar.parse(text)
    return ClauseVisitor().visit(parsed)


print(ClauseVisitor().visit(parsed))
