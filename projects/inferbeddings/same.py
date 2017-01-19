import re
from jtr.projects.inferbeddings.lang import *
import jtr.projects.inferbeddings.engine as engine
import tensorflow as tf

dl = engine.LowRankLog(150, 60, emb_dim=10, reg_pred=1.0, emb_std_dev=1.0,
                       linear_squash=True, trainer=tf.train.AdamOptimizer(learning_rate=0.001))
# dl.add_template_loss(to_template("r(a,b)"), *engine.template_loss_fact_arity_2_complex_neg_sample(dl, scale=1.0))
dl.add_template_loss(to_template("r(a,b)"), *engine.template_loss_fact_arity_2_complex(dl, scale=0.1))
dl.add_template_loss(to_template("!r(X,Y)"), *engine.template_loss_neg_fol_arity_2_complex(dl, 10000))
dl.add_template_query(to_template("r(a,b)"), *engine.template_query_fact_arity_2_complex(dl))

for i in range(0,100):
    dl.add_clause("same(e{},e{})".format(i,i))

dl.add_clause("!same(X,Y)")

dl.learn(10000)

print("------")
print(dl.query("same(e1,e1)"))
print(dl.query("same(e1,e2)"))
print(dl.query("same(a1,a2)"))

print(dl.predicate_embedding("same"))
print(dl.entity_embedding("e1"))
print(dl.entity_embedding("e2"))
print(dl.entity_embedding("a1"))
# print(dl.sess.run(dl.score_arity_2_ids_complex([dl.pred_vocab("same")],[dl.ent_vocab("e1")],[dl.ent_vocab("e1")])))

