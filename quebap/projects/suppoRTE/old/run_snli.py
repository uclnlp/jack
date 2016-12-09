"""
adapted from Dirk Weissenborn's train_snli
"""
import tensorflow as tf
import quebap.projects.suppoRTE.prepare_embeddings as emb 
from quebap.projects.suppoRTE.snli_trainer import train


"""
input parameters
"""

# data loading specifics
tf.app.flags.DEFINE_string('data', './quebap/data/SNLI/snli_1.0/', 'data dir of SNLI.')
tf.app.flags.DEFINE_string('embedding_format', 'pkl', 'pkl|glove|word2vec_bin|word2vec')
tf.app.flags.DEFINE_string('embedding_file', './quebap/data/GloVe/glove.6B.50d.pkl', 'path to embeddings file')

# model
tf.app.flags.DEFINE_integer("mem_size", 200, "hidden size of model")
tf.app.flags.DEFINE_integer("h_size", 200, "size of interaction")

# training
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("l2_lambda", 0, "L2-regularization raten (only for batch training).")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay when loss on validation set does not improve.")
tf.app.flags.DEFINE_integer("batch_size", 50, "Number of examples per batch.")
tf.app.flags.DEFINE_integer("min_epochs", 3, "Minimum num of epochs")
tf.app.flags.DEFINE_string("cell", 'LSTM', "LSTM|GRU")
tf.app.flags.DEFINE_integer("seed", 12345, "Random seed.")
tf.app.flags.DEFINE_integer("runs", 10, "How many runs.")
tf.app.flags.DEFINE_string('embedding_mode', 'fixed', 'fixed|tuned|combined')
#tf.app.flags.DEFINE_integer('tunable_dim', 10,
#                            'number of dims for tunable embeddings if embedding mode is combined')
tf.app.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout.")
tf.app.flags.DEFINE_integer('checkpoint', 1000, 'number of batches until checkpoint.')
#tf.app.flags.DEFINE_integer('num_copies', 1, 'number of copies for associative RNN.')
#tf.app.flags.DEFINE_integer('num_read_keys', 0, 'number of additional read keys for associative RNN.')
tf.app.flags.DEFINE_string("result_file", None, "Where to write results.")
#tf.app.flags.DEFINE_string("moru_ops", 'max,mul,keep,replace,diff,min,forget', "operations of moru cell.")
#tf.app.flags.DEFINE_string("moru_op_biases", None, "biases of moru operations at beginning of training. "
#                                                   "Defaults to 0 for each.")
#tf.app.flags.DEFINE_integer("moru_op_ctr", None, "Size of op ctr. By default ops are controlled by current input"
#                                                 "and previous state. Given a positive integer, an additional"
#                                                 "recurrent op ctr is introduced in MORUCell.")
tf.app.flags.DEFINE_boolean('eval', False, 'only evaluation')
tf.app.flags.DEFINE_string('model_path', './quebap/projects/suppoRTE/save/', 'only evaluation')
#tf.app.flags.DEFINE_string('device', '/gpu:0', 'device to run on')
tf.app.flags.DEFINE_string('device', '/cpu:0', 'device to run on')

FLAGS = tf.app.flags.FLAGS

"""
load data
"""

kwargs = None
if FLAGS.embedding_format == "glove":
    #"wiki-6B" smallest
    kwargs = {"vocab_size": 400000, "dim": 50}
    #"common-crawl-840B"
    #kwargs = {"vocab_size": 2196017, "dim": 300}

print("Loading embeddings...")
embeddings = emb.load(FLAGS.embedding_file, FLAGS.embedding_format)

print("Done.")


import json
print("Configuration: ")
print(json.dumps(FLAGS.__flags, sort_keys=True, indent=2, separators=(',', ': ')))



train(embeddings, FLAGS)







