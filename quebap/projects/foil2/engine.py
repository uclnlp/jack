import tensorflow as tf
import quebap.projects.foil2.lang as lang
from quebap.sisyphos.vocab import Vocab


class LowRankLog:
    """
    An interpreter of low-rank-log programs.
    """

    def __init__(self, max_num_entities, max_num_predicates, emb_dim, reg_pred1=0.00001):
        self.max_num_entities = max_num_entities
        self.max_num_predicates = max_num_predicates
        self.emb_dim = emb_dim
        self.sess = tf.Session()

        self.clauses = []
        self.template_losses = {}
        self.template_queries = {}

        self.ent_embeddings_raw = tf.Variable(tf.random_normal([max_num_entities, emb_dim], stddev=0.1))
        self.ent_embeddings = tf.sigmoid(self.ent_embeddings_raw)
        self.pred1_embeddings = tf.Variable(tf.random_normal([max_num_predicates, emb_dim], stddev=0.1))
        self.total_loss = reg_pred1 * tf.nn.l2_loss(self.pred1_embeddings)
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=0.1)
        self.min_op = None
        self.ent_vocab = Vocab(unk=None)
        self.pred1_vocab = Vocab(unk=None)

    def add_clause(self, clause):
        """
        Add a clause to the current program.
        Args:
            clause: a `Clause` object or a string representing a clause.

        Returns:
            Nothing
        """
        if isinstance(clause, str):
            clause = lang.parse_clause(clause)
        self.clauses.append(clause)
        template = lang.to_template(clause)
        if template in self.template_losses:
            _, add, _ = self.template_losses[template]
            add(clause)
        else:
            print("Warning: template {} not supported".format(template))

    def add_template_loss(self, template, loss, add_clause, get_feed_dict):
        self.template_losses[template] = (loss, add_clause, get_feed_dict)
        self.total_loss = self.total_loss + loss

    def add_template_query(self, template, scores, get_feed_dict):
        self.template_queries[template] = (scores, get_feed_dict)

    def query(self, atom):
        if isinstance(atom, str):
            atom = lang.parse_clause(atom)

        template = lang.to_template(atom)
        scores, get_feed_dict = self.template_queries[template]

        result = self.sess.run(scores, feed_dict=get_feed_dict(atom))
        return result

    def _update_min_op(self):
        if self.min_op is None:
            self.min_op = self.trainer.minimize(self.total_loss)
            self.sess.run(tf.initialize_all_variables())

    def score_arity_1_ids(self, pred_ids, ent_ids):
        ent = tf.gather(self.ent_embeddings, ent_ids)  # [num_facts, emb_dim]
        pred1 = tf.gather(self.pred1_embeddings, pred_ids)  # [num_facts, emb_dim]
        return self.score_arity_1_embs(ent, pred1)

    def score_arity_1_embs(self, ent, pred1):
        scores = tf.reduce_sum(ent * pred1, 1)
        return scores

    def learn(self, num_epochs=100):
        self._update_min_op()

        for epoch in range(0, num_epochs):
            feed_dict = {}
            for _, _, get_feed_dict in self.template_losses.values():
                feed_dict.update(get_feed_dict())

            loss, _ = self.sess.run([self.total_loss, self.min_op], feed_dict=feed_dict)
            if epoch % (num_epochs / 5) == 0:
                print(loss)
                # print(self.query("lecturer(seb)"))
                # print(self.sess.run(tf.gradients(loss, self.ent_embeddings),feed_dict=feed_dict))


def template_loss_fact_arity_1(engine: LowRankLog):
    clauses = []

    ph_ids_pred = tf.placeholder(tf.int32, [None])
    ph_ids_ent = tf.placeholder(tf.int32, [None])

    ids_pred = []
    ids_ent = []

    def add_new_clause(clause):
        ids_pred.append(engine.pred1_vocab(clause.head.predicate))
        ids_ent.append(engine.ent_vocab(clause.head.arguments[0]))
        clauses.append(clause)

    def get_feed_dict():
        return {
            ph_ids_pred: ids_pred,
            ph_ids_ent: ids_ent
        }

    scores = engine.score_arity_1_ids(ph_ids_pred, ph_ids_ent)
    losses = tf.maximum(-scores + 1, 0)  # hinge loss with margin 1, [num_facts]
    loss = tf.reduce_sum(losses)

    return loss, add_new_clause, get_feed_dict


def template_loss_neg_fol_arity_1(engine: LowRankLog, num_samples=10):
    ph_ids_pred = tf.placeholder(tf.int32, [None])
    ids_pred = []

    def add_new_clause(clause):
        ids_pred.append(engine.pred1_vocab(clause.head.predicate))

    def get_feed_dict():
        return {
            ph_ids_pred: ids_pred,
        }

    pred1 = tf.gather(engine.pred1_embeddings, ph_ids_pred)  # [num_clauses, emb_dim]

    # sample a set of random entity embeddings, uniformly
    sampled_ent_embs = tf.random_uniform((num_samples, engine.emb_dim), 0.0, 1.0)  # [num_samples, emb_dim]

    scores = tf.matmul(sampled_ent_embs, pred1, transpose_b=True)  # [num_samples, num_clauses]

    losses = tf.maximum(scores + 1, 0)
    loss = tf.reduce_mean(losses)  # / num_samples

    return loss, add_new_clause, get_feed_dict


def template_query_fact_arity_1(engine):
    ph_ids_pred = tf.placeholder(tf.int32, [None])
    ph_ids_ent = tf.placeholder(tf.int32, [None])

    def get_feed_dict(clause):
        id_pred = engine.pred1_vocab(clause.head.predicate)
        id_ent = engine.ent_vocab(clause.head.arguments[0])
        return {
            ph_ids_ent: (id_ent,),
            ph_ids_pred: (id_pred,)
        }

    scores = engine.score_arity_1_ids(ph_ids_pred, ph_ids_ent)[0]

    return scores, get_feed_dict


dl = LowRankLog(10, 10, 3, reg_pred1=0.0001)
dl.add_template_loss(lang.to_template("r(a)"), *template_loss_fact_arity_1(dl))
dl.add_template_loss(lang.to_template("!r(X)"), *template_loss_neg_fol_arity_1(dl, 10000))

dl.add_template_query(lang.to_template("r(a)"), *template_query_fact_arity_1(dl))

dl.add_clause("lecturer(seb)")
dl.add_clause("employee(seb)")
dl.add_clause("artist(mika)")
dl.add_clause("lecturer(iasonas)")
dl.add_clause("!artist(X)")
dl.add_clause("!lecturer(X)")
dl.add_clause("!employee(X)")

dl.learn(10000)

print("---")
print(dl.query("lecturer(seb)"))
print(dl.query("lecturer(iasonas)"))
print(dl.query("employee(iasonas)"))
print(dl.query("artist(iasonas)"))
print("---")

print(dl.query("artist(seb)"))
print(dl.query("artist(mika)"))
print(dl.query("lecturer(mika)"))
print(dl.query("employee(mika)"))

