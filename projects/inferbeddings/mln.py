import re
from jtr.projects.inferbeddings.lang import *
import jtr.projects.inferbeddings.engine as engine
import tensorflow as tf


def load_mln_db_file(filename):
    result = []
    with open(filename) as db_file:
        for line in db_file:
            split = [t for t in re.split('\(|\)|,|\n', line) if t != '']
            if len(split) == 3:
                pred, arg1, arg2 = split
                result.append(Clause(Atom(Predicate(pred), Constant(arg1), Constant(arg2))))
            elif len(split) == 4:
                pred, arg1, arg2, arg3 = split
                result.append(
                    Clause(Atom(Predicate(pred + "_" + arg1), Constant(arg2), Constant(arg3))))
    return result


facts = load_mln_db_file("/Users/riedel/corpora/mln/nat.db")
preds = set(f.head.predicate for f in facts)
# print("\n".join(str(c) for c in facts))

dl = engine.LowRankLog(150, 60, emb_dim=6, reg_pred=0.0, emb_std_dev=1.0,
                       linear_squash=False, trainer=tf.train.AdamOptimizer(learning_rate=0.01))
dl.add_template_loss(to_template("r(a,b)"), *engine.template_loss_fact_arity_2_complex(dl, scale=0.0000001))
# dl.add_template_loss(to_template("!r(X)"), *engine.template_loss_neg_fol_arity_1(dl, 10000))
dl.add_template_loss(to_template("!r(X,Y)"), *engine.template_loss_neg_fol_arity_2_complex(dl, 10))
# dl.add_template_loss(to_template("r(X,Y):-q(Y,X)"),
#                      *engine.template_loss_imply_arity_2_same_entities(dl, 10000, switch_args=True))
# dl.add_template_loss(to_template("r(X,Y):-q(X,Y)"),
#                      *engine.template_loss_imply_arity_2_same_entities(dl, 10000, switch_args=False))

# dl.add_template_query(to_template("r(a)"), *engine.template_query_fact_arity_1(dl))
dl.add_template_query(to_template("r(a,b)"), *engine.template_query_fact_arity_2_complex(dl))

print(len(facts))
for fact in facts[:100]:
    dl.add_clause(fact)

# dl.add_clause(Clause(Atom(Predicate("R"), Constant("China"), Constant("Area"))))

for pred in preds:  # [Predicate("R")]: #list(preds)[:1]:
    dl.add_clause(Clause(Atom(pred, Variable("X"), Variable("Y"), negated=True)))

dl.learn(10000)

print("---")
print(dl.query(facts[0]))
print(dl.query(Clause(Atom(Predicate("R"), Constant("China"), Constant("Area")))))
print(dl.query(Clause(Atom(Predicate("R"), Constant("China"), Constant("Divorces")))))
print(dl.query(Clause(Atom(Predicate("R"), Constant("China"), Constant("Catholics")))))
print(dl.query(Clause(Atom(Predicate("R"), Constant("Chasina"), Constant("Divorces")))))
print(dl.query(Clause(Atom(Predicate("R"), Constant("Chasina"), Constant("Dasdivorces")))))
print(dl.query(Clause(Atom(Predicate("R"), Constant("India"), Constant("Divorces")))))
print(dl.query(Clause(Atom(Predicate("R"), Constant("China"), Constant("Usaidreceived")))))
print(dl.query(Clause(Atom(Predicate("R"), Constant("China"), Constant("Roadlength")))))
print(dl.query(Clause(Atom(Predicate("R"), Constant("China"), Constant("Noncommunist")))))
print(dl.query(Clause(Atom(Predicate("R"), Constant("Burma"), Constant("Roadlength")))))
print(dl.query(Clause(Atom(Predicate("R"), Constant("Netherlands"), Constant("Usaidreceived")))))
