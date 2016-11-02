class Expr:
    def __repr__(self):
        return self.__dict__.__repr__()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and str(other) == str(self)

    def __hash__(self):
        return self.__str__().__hash__()


class Variable(Expr):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class Constant(Expr):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class Predicate(Expr):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, self.__class__) and other.name == self.name

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return self.name


class Atom(Expr):
    def __init__(self, predicate: Predicate, *arguments, negated=False):
        self.predicate = predicate
        self.arguments = arguments
        self.negated = negated

    def __repr__(self):
        return "{}{}({})".format("!" if self.negated else "",
                                 self.predicate.name,
                                 ", ".join(str(x) for x in self.arguments))


class Clause(Expr):
    def __init__(self, head: Atom, *body):
        self.head = head
        self.body = body

    def __repr__(self):
        return str(self.head) if len(self.body) == 0 else "{head} :- {body}".format(
            head=self.head,
            body=", ".join(str(x) for x in self.body))


from parsimonious.grammar import Grammar, NodeVisitor

grammar = Grammar(
    """
    clause     = atom ( ":-" _ atom_list)?
    atom_list  = atom  ("," _ atom_list)? _
    atom       = neg? predicate "(" _ term_list ")" _
    neg        = "!" _
    term_list  = term  ("," _ term_list)? _
    term       = constant / variable
    predicate  = low_id / string
    constant   = low_id / string
    variable   = ~"[A-Z][a-z A-Z 0-9_]*"
    low_id     = ~"[a-z][a-z A-Z 0-9_]*"
    string     = ~r"'[^']*'" / ~r"\\"[^\\"]*\\""
    _          = skip*
    skip       = ~r"\s+"
    """)


class ClauseVisitor(NodeVisitor):
    def visit_clause(self, node, visited_children):
        # print("Clause")
        if len(visited_children[1]) == 0:
            return Clause(visited_children[0])
        else:
            # print(visited_children)
            head, ((_, _, body),) = visited_children
            return Clause(head, *body)

    def visit_predicate(self, _, visited_children):
        return Predicate(visited_children[0])

    def visit_constant(self, _, visited_children):
        return Constant(visited_children[0])

    def visit_variable(self, node, _):
        return Variable(node.full_text[node.start:node.end])

    def visit_term(self, _, visited_children):
        # print("In Term")
        # print(visited_children)
        return visited_children[0]

    def visit_term_list(self, _, visited_children):
        # print("TermList")
        # print(visited_children)
        if len(visited_children[1]) == 0:
            return visited_children[:1]
        else:
            head, ((_, _, tail),), _ = visited_children
            return [head] + tail

    def visit_atom_list(self, _, visited_children):
        # print("AtomList")
        # print(visited_children)
        if len(visited_children[1]) == 0:
            return visited_children[:1]
        else:
            # print(visited_children)
            head, ((_, _, tail),), _ = visited_children
            return [head] + tail

    def visit__(self, _, visited_children):
        # print("____")
        return []

    def visit_atom(self, _, visited_children):
        # print("In Atom")
        # print(visited_children)
        return Atom(visited_children[1], *visited_children[4], negated=len(visited_children[0]) == 1)

    def visit_low_id(self, node, _):
        return node.full_text[node.start:node.end]

    def visit_up_id(self, node, _):
        return node.full_text[node.start:node.end]

    def visit_string(self, node, _):
        return node.full_text[node.start:node.end]

    def generic_visit(self, node, visited_children):
        return visited_children

    def visit_neg(self, _1, _2):
        return True


def parse_clause(text):
    parsed = grammar.parse(text)
    return ClauseVisitor().visit(parsed)


from collections import defaultdict


def to_template(expr):
    if isinstance(expr, str):
        expr = parse_clause(expr)
    predicates = {}
    variables = {}
    constants = {}

    def get_or_else_inc(dictionary, prefix, key, always_inc=False):
        if key in dictionary and not always_inc:
            return dictionary[key]
        else:
            dictionary[key] = prefix + str(len(dictionary))
            return dictionary[key]

    def recurse(recursed_expr):

        if isinstance(recursed_expr, Clause):
            return Clause(recurse(recursed_expr.head), *(recurse(a) for a in recursed_expr.body))
        elif isinstance(recursed_expr, Atom):
            return Atom(Predicate(get_or_else_inc(predicates, "_pred_", recursed_expr.predicate)),
                        *(recurse(arg) for arg in recursed_expr.arguments),
                        negated=recursed_expr.negated)
        elif isinstance(recursed_expr, Constant):
            return Constant(get_or_else_inc(constants, "_constant_", recursed_expr.name, True))
        elif isinstance(recursed_expr, Variable):
            return Constant(get_or_else_inc(variables, "_variables_", recursed_expr.name))

    return recurse(expr)


if __name__ == '__main__':
    # print(parse_clause("test(a,b)"))
    # print(parse_clause("r(X,Y) :- q(Y,X)"))
    # print(parse_clause("r(X,Y) :- q(Y,X), t(X)"))

    print(parse_clause("! r(a)"))
    print(to_template(parse_clause("r(X,a):-q(a,X)")))

    print(parse_clause("r(X,a):-q(Y,X)") == parse_clause("r(X,a):-q(Y,X)"))

    test = {parse_clause("r(X,a):-q(Y,X)"): 5}

    print(test[parse_clause("r(X,a):-q(Y,X)")])
