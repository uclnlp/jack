# -*- coding: utf-8 -*-

# coding=utf-8
"""
         __  _ __
  __  __/ /_(_) /
 / / / / __/ / /
/ /_/ / /_/ / /
\__,_/\__/_/_/ v0.1

Making useful stuff happen since 2016
"""

import tensorflow as tf

"""
### TF utils ###
"""


def tfrun(tensor):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        return sess.run(tensor)


def tfrunprint(tensor, suffix="", prefix=""):
    if prefix == "":
        print(tfrun(tensor), suffix)
    else:
        print(prefix, tfrun(tensor), suffix)


def tfrunprintshape(tensor, suffix="", prefix=""):
    tfrunprint(tf.shape(tensor), suffix, prefix)


def tfprint(tensor, fun=None, prefix=""):
    if fun is None:
        fun = lambda x: x
    return tf.Print(tensor, [fun(tensor)], prefix)


def tfprints(tensors, fun=None, prefix=""):
    if fun is None:
        fun = lambda x: x
    prints = []
    for i in range(0, len(tensors)):
        prints.append(tf.Print(tensors[i], [fun(tensors[i])], prefix))
    return prints


def tfprintshapes(tensors, prefix=""):
    return tfprints(tensors, lambda x: tf.shape(x), prefix)


def tfprintshape(tensor, prefix=""):
    return tfprint(tensor, lambda x: tf.shape(x), prefix)


def gather_in_dim(params, indices, dim, name=None):
    """
    Gathers slices in a defined dimension. If dim == 0 this is doing the same
      thing as tf.gather.
    """
    if dim == 0:
        return tf.gather(params, indices, name)
    else:
        dims = [i for i in range(0, len(params.get_shape()))]
        to_dims = list(dims)
        to_dims[0] = dim
        to_dims[dim] = 0

        transposed = tf.transpose(params, to_dims)
        gathered = tf.gather(transposed, indices)
        reverted = tf.transpose(gathered, to_dims)

        return reverted


def unit_length(tensor):
    l2norm_sq = tf.reduce_sum(tensor * tensor, 1, keep_dims=True)
    l2norm = tf.rsqrt(l2norm_sq)
    return tensor * l2norm


"""
### NTP utils ###
"""


def head(rule):
    return rule[0]


def body(rule):
    return rule[1:]


def isvar(sym):
    """
    Checks whether a symbol in a structure is a variable, e.g., X,Y,Z etc.
    """
    if isinstance(sym, str):
        return sym.isupper()
    else:
        return False


def isplaceholder(sym):
    """
    Checks whether a symbol in a structure is a placeholder for a representation
    """
    if isinstance(sym, str):
        return sym[0] == "#"
    else:
        return False


def isconst(sym):
    """
    Checks whether a symbol is a constant or predicate
    """
    return not (isvar(sym) or isplaceholder(sym))


def isparam(sym):
    """
    Checks whether a symbol represents a rule parameter
    """
    return isinstance(sym, str) and sym.startswith("param_")


def get_param_id(sym):
    return int(sym[6:])


def isfact(rule):
    """
    Checks whether rule represents a fact, i.e.,
    body is empty and atom is grounded
    """
    return len(body(rule)) == 0 and all(isconst(sym) for sym in head(rule))


def placeholder2ix(placeholder):
    """
    Extracts index from placeholder, e.g., #₂ => 2
    """
    return int(placeholder[1:])


def sub_struct2str(sub_struct):
    return "{" + ", ".join([X1 + "/" + X2 for (X1, X2) in sub_struct]) + "}"


def atom2str(atom):
    return str(atom[0]) + "(" + ", ".join([str(x) for x in atom[1:]]) + ")"


def rule2str(rule):
    return " ∧ ".join([atom2str(atom) for atom in body(rule)]) + " ⇒ " + \
           atom2str(head(rule))


def atom2struct(atom):
    """
    :param atom: an atom such as [parentOf, X, Y]
    :return: structure of the atom, e.g., (#₁, X, Y)
    """
    struct = []
    counter = 0

    for sym in atom:
        if isconst(sym):
            struct.append("#" + str(counter))
            counter += 1
        else:
            struct.append(sym)

    return tuple(struct)


# todo: symbols in different atoms can currently have the same placeholder
# is this intended?
def rule2struct(rule):
    return tuple([atom2struct(atom) for atom in rule])


def rule2tuple(rule):
    return tuple([tuple(a) for a in rule])


def normalize_rule(rule):
    """
    Renames variables in rule to maximize structural similarity to other rules.
    :param rule: a rule, such as
      [[grandparentOf, X, Z], [parentOf, X, Y], [parentOf, Y, Z]]
    :return: standardized rule, e.g.,
      [[grandparentOf, X₀, X₁], [parentOf, X₀, X₂], [parentOf, X₂, X₁]]
    """
    sub = {}
    counter = 0

    for atom in rule:
        for i in range(0, len(atom)):
            sym = atom[i]
            if isvar(sym):
                if sym not in sub:
                    sub[sym] = "X" + str(counter)
                    counter += 1
                atom[i] = sub[sym]
            else:
                atom[i] = sym

    return rule
