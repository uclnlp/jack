import tensorflow as tf
import quebap.projects.inferbeddings.lang as lang
from quebap.sisyphos.vocab import Vocab


class LowRankLog:
    """
    An interpreter of low-rank-log programs.
    """

    def __init__(self, max_num_entities, max_num_predicates, emb_dim,
                 reg_pred=0.00001, emb_std_dev=0.1, linear_squash=False,
                 trainer=tf.train.RMSPropOptimizer(learning_rate=0.1)):
        self.max_num_entities = max_num_entities
        self.max_num_predicates = max_num_predicates
        self.emb_dim = emb_dim
        self.sess = tf.Session()

        self.clauses = []
        self.template_losses = {}
        self.template_queries = {}
        self.active_templates = set()

        if linear_squash:
            self.ent_embeddings_raw = tf.Variable(
                tf.random_uniform([max_num_entities, emb_dim], minval=0.0, maxval=1.0))
            self.ent_embeddings = tf.minimum(1.0, tf.maximum(0.0, self.ent_embeddings_raw))
        else:
            self.ent_embeddings_raw = tf.Variable(tf.random_normal([max_num_entities, emb_dim], stddev=emb_std_dev))
            self.ent_embeddings = tf.sigmoid(self.ent_embeddings_raw)

        self.pred_embeddings = tf.Variable(tf.random_normal([max_num_predicates, emb_dim], stddev=emb_std_dev))
        self.total_loss = reg_pred * tf.nn.l2_loss(self.pred_embeddings)
        self.trainer = trainer
        self.min_op = None
        self.ent_vocab = Vocab(unk=None)
        self.pred_vocab = Vocab(unk=None)

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
            loss, add, _ = self.template_losses[template]
            add(clause)
            if template not in self.active_templates:
                self.active_templates.add(template)
                self.total_loss = self.total_loss + loss

        else:
            print("Warning: template {} not supported".format(template))

    def add_template_loss(self, template, loss, add_clause, get_feed_dict):
        if isinstance(template, str):
            template = lang.to_template(template)
        self.template_losses[template] = (loss, add_clause, get_feed_dict)

    def add_template_query(self, template, scores, get_feed_dict):
        if isinstance(template, str):
            template = lang.to_template(template)
        self.template_queries[template] = (scores, get_feed_dict)

    def entity_embedding(self, entity_name):
        return self.sess.run(tf.gather(self.ent_embeddings, self.ent_vocab.get_id(entity_name)))

    def predicate_embedding(self, predicate_name):
        return self.sess.run(tf.gather(self.pred_embeddings, self.pred_vocab.get_id(predicate_name)))

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
        pred1 = tf.gather(self.pred_embeddings, pred_ids)  # [num_facts, emb_dim]
        return self.score_arity_1_embs(ent, pred1)

    def score_arity_1_embs(self, ent, pred1):
        scores = tf.reduce_sum(ent * pred1, 1)
        return scores

    def score_arity_2_ids_complex(self, pred_ids, ent1_ids, ent2_ids):
        ent1 = tf.gather(self.ent_embeddings, ent1_ids)  # [num_facts, emb_dim]
        ent2 = tf.gather(self.ent_embeddings, ent2_ids)  # [num_facts, emb_dim]
        pred1 = tf.gather(self.pred_embeddings, pred_ids)  # [num_facts, emb_dim]
        return self.score_arity_2_embs_complex(pred1, ent1, ent2)

    def score_arity_2_embs_complex(self, pred2, ent1, ent2):
        # pred2: [num_samples, dim]
        # ent1: [num_samples, dim]
        # ent2: [num_samples, dim]
        # def dot3(arg1, rel, arg2):
        #     return tf.matmul(arg1 * arg2, rel, transpose_b=True)  # [num_samples, num_preds]
        def dot3(arg1, rel, arg2):
            return tf.reduce_sum(arg1 * rel * arg2, 1)

        ent1_re = ent1[:, :self.emb_dim // 2]
        ent1_im = ent1[:, self.emb_dim // 2:]
        ent2_re = ent2[:, :self.emb_dim // 2]
        ent2_im = ent2[:, self.emb_dim // 2:]
        pred2_re = pred2[:, :self.emb_dim // 2]
        pred2_im = pred2[:, self.emb_dim // 2:]

        # scores = tf.reduce_sum(ent1_re * pred2_re * ent2_re, 1) + \
        #          tf.reduce_sum(ent1_re * pred2_im * ent2_im, 1) + \
        #          tf.reduce_sum(ent1_im * pred2_re * ent2_im, 1) - \
        #          tf.reduce_sum(ent1_im * pred2_im * ent2_re, 1)
        scores = dot3(ent1_re, pred2_re, ent2_re) + \
                 dot3(ent1_re, pred2_im, ent2_im) + \
                 dot3(ent1_im, pred2_re, ent2_im) - \
                 dot3(ent1_im, pred2_im, ent2_re)

        return scores

    def score_arity_2_embs_complex_tiled(self, pred2, ent1, ent2):
        # pred2: [num_clauses, num_samples, dim]
        # ent1: [num_clauses, num_samples, dim]
        # ent2: [num_clauses, num_samples, dim]

        # def dot3(arg1, rel, arg2):
        #     return tf.matmul(arg1 * arg2, rel, transpose_b=True)  # [num_samples, num_preds]
        def dot3(arg1, rel, arg2):
            return tf.reduce_sum(arg1 * rel * arg2, 2)

            # return arg1 + rel + arg2 # tf.reduce_sum(arg1 + rel + arg2, 2)

        ent1_re = ent1[:, :, :self.emb_dim // 2]
        ent1_im = ent1[:, :, self.emb_dim // 2:]
        ent2_re = ent2[:, :, :self.emb_dim // 2]
        ent2_im = ent2[:, :, self.emb_dim // 2:]
        pred2_re = pred2[:, :, :self.emb_dim // 2]
        pred2_im = pred2[:, :, self.emb_dim // 2:]

        # scores = tf.reduce_sum(ent1_re * pred2_re * ent2_re, 1) + \
        #          tf.reduce_sum(ent1_re * pred2_im * ent2_im, 1) + \
        #          tf.reduce_sum(ent1_im * pred2_re * ent2_im, 1) - \
        #          tf.reduce_sum(ent1_im * pred2_im * ent2_re, 1)
        scores = dot3(ent1_re, pred2_re, ent2_re) + \
                 dot3(ent1_re, pred2_im, ent2_im) + \
                 dot3(ent1_im, pred2_re, ent2_im) - \
                 dot3(ent1_im, pred2_im, ent2_re)

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
                # print(self.sess.run(tf.shape(loss)))

                # print(self.query("livesin(seb,london)"))
                # print(self.sess.run(tf.gradients(loss, self.ent_embeddings),feed_dict=feed_dict))


def template_loss_fact_arity_1(engine: LowRankLog):
    clauses = []

    ph_ids_pred = tf.placeholder(tf.int32, [None])
    ph_ids_ent = tf.placeholder(tf.int32, [None])

    ids_pred = []
    ids_ent = []

    def add_new_clause(clause):
        ids_pred.append(engine.pred_vocab(clause.head.predicate))
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


def only_if_no_ids(term, ids):
    return tf.case([(tf.greater(tf.shape(ids)[0], 0), lambda: term)], default=lambda: tf.constant(0.0))


def template_loss_fact_arity_2_complex(engine: LowRankLog, scale=1.0):
    clauses = []

    ph_ids_pred = tf.placeholder(tf.int32, [None], name="pred2")
    ph_ids_ent1 = tf.placeholder(tf.int32, [None], name="ent1")
    ph_ids_ent2 = tf.placeholder(tf.int32, [None], name="ent2")

    ids_pred = []
    ids_ent1 = []
    ids_ent2 = []

    def add_new_clause(clause):
        ids_pred.append(engine.pred_vocab(clause.head.predicate))
        ids_ent1.append(engine.ent_vocab(clause.head.arguments[0]))
        ids_ent2.append(engine.ent_vocab(clause.head.arguments[1]))
        clauses.append(clause)

    def get_feed_dict():
        return {
            ph_ids_pred: ids_pred,
            ph_ids_ent1: ids_ent1,
            ph_ids_ent2: ids_ent2
        }

    scores = engine.score_arity_2_ids_complex(ph_ids_pred, ph_ids_ent1, ph_ids_ent2)
    losses = tf.maximum(-scores + 1, 0)  # hinge loss with margin 1, [num_facts]
    loss = scale * tf.reduce_sum(losses)
    # loss = scale * tf.reduce_mean(losses)
    # tf.Print(loss, [scores], "Facts:")
    # return tf.Print(loss, [loss, losses, scores], "Facts:"), add_new_clause, get_feed_dict
    return loss, add_new_clause, get_feed_dict


def template_loss_fact_arity_2_complex_neg_sample(engine: LowRankLog, scale=1.0):
    clauses = []

    ph_ids_pred = tf.placeholder(tf.int32, [None], name="pred2")
    ph_ids_ent1 = tf.placeholder(tf.int32, [None], name="ent1")
    ph_ids_ent2 = tf.placeholder(tf.int32, [None], name="ent2")

    ids_pred = []
    ids_ent1 = []
    ids_ent2 = []

    def add_new_clause(clause):
        ids_pred.append(engine.pred_vocab(clause.head.predicate))
        ids_ent1.append(engine.ent_vocab(clause.head.arguments[0]))
        ids_ent2.append(engine.ent_vocab(clause.head.arguments[1]))
        clauses.append(clause)

    def get_feed_dict():
        return {
            ph_ids_pred: ids_pred,
            ph_ids_ent1: ids_ent1,
            ph_ids_ent2: ids_ent2
        }

    pos_scores = engine.score_arity_2_ids_complex(ph_ids_pred, ph_ids_ent1, ph_ids_ent2)
    pos_losses = tf.maximum(-pos_scores + 1, 0)  # hinge loss with margin 1, [num_facts]
    pos_loss = scale * tf.reduce_sum(pos_losses)
    # loss = scale * tf.reduce_mean(losses)
    # tf.Print(loss, [scores], "Facts:")
    # return tf.Print(loss, [loss, losses, scores], "Facts:"), add_new_clause, get_feed_dict
    neg_scores = engine.score_arity_2_ids_complex(tf.random_shuffle(ph_ids_pred),
                                                  tf.random_shuffle(ph_ids_ent1),
                                                  tf.random_shuffle(ph_ids_ent2))
    neg_losses = tf.maximum(neg_scores + 1, 0)  # hinge loss with margin 1, [num_facts]
    neg_loss = scale * tf.reduce_sum(neg_losses)

    loss = pos_loss + neg_loss

    return loss, add_new_clause, get_feed_dict


def template_loss_neg_fol_arity_1(engine: LowRankLog, num_samples=10):
    ph_ids_pred = tf.placeholder(tf.int32, [None])
    ids_pred = []

    def add_new_clause(clause):
        ids_pred.append(engine.pred_vocab(clause.head.predicate))

    def get_feed_dict():
        return {
            ph_ids_pred: ids_pred,
        }

    pred1 = tf.gather(engine.pred_embeddings, ph_ids_pred)  # [num_clauses, emb_dim]

    # sample a set of random entity embeddings, uniformly
    sampled_ent_embs = tf.random_uniform((num_samples, engine.emb_dim), 0.0, 1.0)  # [num_samples, emb_dim]

    scores = tf.matmul(sampled_ent_embs, pred1, transpose_b=True)  # [num_samples, num_clauses]

    losses = tf.maximum(scores + 1, 0)
    loss = tf.reduce_sum(tf.reduce_mean(losses, 0))  # / num_samples

    return loss, add_new_clause, get_feed_dict


def template_loss_neg_fol_arity_2_complex(engine: LowRankLog, num_samples=10):
    ph_ids_pred = tf.placeholder(tf.int32, [None])
    ids_pred = []

    def add_new_clause(clause):
        ids_pred.append(engine.pred_vocab(clause.head.predicate))

    def get_feed_dict():
        return {
            ph_ids_pred: ids_pred,
        }

    pred1 = tf.expand_dims(tf.gather(engine.pred_embeddings, ph_ids_pred), 1)  # [num_clauses, 1, emb_dim]

    # sample a set of random entity embeddings, uniformly
    # [1, num_samples, emb_dim]
    sampled_ent1_embs = tf.expand_dims(tf.random_uniform((num_samples, engine.emb_dim), 0.0, 1.0), 0)
    sampled_ent2_embs = tf.expand_dims(tf.random_uniform((num_samples, engine.emb_dim), 0.0, 1.0), 0)

    # need to tile with num_clauses
    # pred = [num_clauses, 1, dim]
    # arg2 = [1, num_samples, dim]

    # scores = tf.matmul(sampled_ent1_embs, pred1, transpose_b=True)  # [num_samples, num_clauses]
    scores = engine.score_arity_2_embs_complex_tiled(pred1, sampled_ent1_embs,
                                                     sampled_ent2_embs)  # [num_clauses,num_samples]
    # scores = tf.constant([[1.0]])  # [num_clauses,num_samples]

    losses = tf.maximum(scores + 1, 0)
    loss = tf.reduce_sum(tf.reduce_mean(losses, 1), 0)  # / num_samples

    return loss, add_new_clause, get_feed_dict
    # return tf.Print(loss,[loss,tf.reduce_max(losses,[0,1]),scores],"Neg: ",summarize=10), add_new_clause, get_feed_dict


def template_loss_imply_arity_2_same_entities(engine: LowRankLog, num_samples=10,
                                              body_neg=False, head_neg=False, switch_args=False):
    ph_ids_pred_pred = tf.placeholder(tf.int32, [None])
    ph_ids_body_pred = tf.placeholder(tf.int32, [None])
    ids_head_pred = []
    ids_body_pred = []

    def add_new_clause(clause: lang.Clause):
        ids_head_pred.append(engine.pred_vocab(clause.head.predicate))
        ids_body_pred.append(engine.pred_vocab(clause.body[0].predicate))

    def get_feed_dict():
        return {
            ph_ids_pred_pred: ids_head_pred,
            ph_ids_body_pred: ids_body_pred,
        }

    pred_head = tf.gather(engine.pred_embeddings, ph_ids_pred_pred)  # [num_clauses, emb_dim]
    pred_body = tf.gather(engine.pred_embeddings, ph_ids_body_pred)  # [num_clauses, emb_dim]

    # sample a set of random entity embeddings, uniformly
    sampled_ent1_embs = tf.random_uniform((num_samples, engine.emb_dim), 0.0, 1.0)  # [num_samples, emb_dim]
    sampled_ent2_embs = tf.random_uniform((num_samples, engine.emb_dim), 0.0, 1.0)  # [num_samples, emb_dim]

    arg1_head, arg2_head = sampled_ent1_embs, sampled_ent2_embs
    arg1_body, arg2_body = (arg1_head, arg2_head) if not switch_args else (arg2_head, arg1_head)

    # scores = tf.matmul(sampled_ent1_embs, pred1, transpose_b=True)  # [num_samples, num_clauses]
    scores_head = engine.score_arity_2_embs_complex(pred_head, arg1_head, arg2_head)
    scores_body = engine.score_arity_2_embs_complex(pred_body, arg1_body, arg2_body)

    scaled_head_scores = -scores_head if head_neg else scores_head
    scaled_body_scores = -scores_body if body_neg else scores_body

    losses = tf.maximum(scaled_body_scores - scaled_head_scores + 1, 0)
    loss = tf.reduce_sum(tf.reduce_mean(losses, 0))  # / num_samples

    return loss, add_new_clause, get_feed_dict


def template_query_fact_arity_1(engine):
    ph_ids_pred = tf.placeholder(tf.int32, [None])
    ph_ids_ent = tf.placeholder(tf.int32, [None])

    def get_feed_dict(clause):
        id_pred = engine.pred_vocab(clause.head.predicate)
        id_ent = engine.ent_vocab(clause.head.arguments[0])
        return {
            ph_ids_ent: (id_ent,),
            ph_ids_pred: (id_pred,)
        }

    scores = engine.score_arity_1_ids(ph_ids_pred, ph_ids_ent)[0]

    return scores, get_feed_dict


def template_query_fact_arity_2_complex(engine):
    ph_ids_pred = tf.placeholder(tf.int32, [None])
    ph_ids_ent1 = tf.placeholder(tf.int32, [None])
    ph_ids_ent2 = tf.placeholder(tf.int32, [None])

    def get_feed_dict(clause):
        id_pred = engine.pred_vocab(clause.head.predicate)
        id_ent1 = engine.ent_vocab(clause.head.arguments[0])
        id_ent2 = engine.ent_vocab(clause.head.arguments[1])
        return {
            ph_ids_ent1: (id_ent1,),
            ph_ids_ent2: (id_ent2,),
            ph_ids_pred: (id_pred,),
        }

    scores = engine.score_arity_2_ids_complex(ph_ids_pred, ph_ids_ent1, ph_ids_ent2)[0]

    return scores, get_feed_dict


def create_inferbeddings(*args, **kwargs):
    dl = LowRankLog(*args, **kwargs)
    dl.add_template_loss(lang.to_template("r(a)"), *template_loss_fact_arity_1(dl))
    dl.add_template_loss(lang.to_template("r(a,b)"), *template_loss_fact_arity_2_complex(dl))
    dl.add_template_loss(lang.to_template("!r(X)"), *template_loss_neg_fol_arity_1(dl, 10000))
    dl.add_template_loss(lang.to_template("!r(X,Y)"), *template_loss_neg_fol_arity_2_complex(dl, 10000))
    dl.add_template_loss(lang.to_template("r(X,Y):-q(Y,X)"),
                         *template_loss_imply_arity_2_same_entities(dl, 10000, switch_args=True))
    dl.add_template_loss(lang.to_template("r(X,Y):-q(X,Y)"),
                         *template_loss_imply_arity_2_same_entities(dl, 10000, switch_args=False))

    dl.add_template_query(lang.to_template("r(a)"), *template_query_fact_arity_1(dl))
    dl.add_template_query(lang.to_template("r(a,b)"), *template_query_fact_arity_2_complex(dl))
    return dl


if __name__ == '__main__':
    dl = create_inferbeddings(10, 10, 10, reg_pred=0.0001)

    dl.add_clause("lecturer(seb)")
    dl.add_clause("employee(seb)")
    dl.add_clause("artist(mika)")
    dl.add_clause("lecturer(iasonas)")
    dl.add_clause("!artist(X)")
    dl.add_clause("!lecturer(X)")
    dl.add_clause("!employee(X)")
    dl.add_clause("livesin(seb,london)")
    dl.add_clause("!livesin(X,Y)")
    # dl.add_clause("!place_of(X,Y)")
    dl.add_clause("place_of(X,Y) :- livesin(Y,X)")

    dl.learn(1000)

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

    print("---")
    print(dl.query("livesin(seb,london)"))
    print(dl.query("livesin(iasonas,london)"))
    print(dl.query("livesin(mika,london)"))
    print(dl.query("livesin(london,seb)"))
    print(dl.query("place_of(london,seb)"))
