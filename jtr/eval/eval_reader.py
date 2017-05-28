import json

import tensorflow as tf

from jtr.jack.data_structures import convert2qasettings
from jtr.jack.readers import readers, eval_hooks
from jtr.io.embeddings import load_embeddings
from jtr.preprocess.vocab import Vocab

tf.app.flags.DEFINE_string('file', None, 'JTR dataset file')
tf.app.flags.DEFINE_string('model', None, 'Name of the reader')
tf.app.flags.DEFINE_string('model_dir', None, 'directory to saved model')
tf.app.flags.DEFINE_string('embedding_path', None, 'path to embeddings')
tf.app.flags.DEFINE_string('embedding_format', 'glove', 'embeddings format')
tf.app.flags.DEFINE_string('device', "/cpu:0", 'device to use')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('beam_size', 1, 'beam size')
tf.app.flags.DEFINE_string('kwargs', '{}', 'additional reader-specific configurations')

FLAGS = tf.app.flags.FLAGS

print("Loading embeddings from %s..." % FLAGS.embedding_path)
emb = load_embeddings(FLAGS.embedding_path, FLAGS.embedding_format)
vocab = Vocab(emb=emb, init_from_embeddings=True)

print("Creating and loading reader from %s..." % FLAGS.model_dir)
config = {"beam_size": FLAGS.beam_size, 'batch_size': FLAGS.batch_size, "max_support_length": None}
config.update(json.loads(FLAGS.kwargs))
reader = readers[FLAGS.model](vocab, config)
with tf.device(FLAGS.device):
    reader.setup_from_file(FLAGS.model_dir)

with open(FLAGS.file) as f:
    dataset_jtr = json.load(f)

dataset = convert2qasettings(dataset_jtr)

print("Start!")


def side_effect(metrics, prev_metric):
    """Returns: a state (in this case a metric) that is used as input for the next call"""
    print("#####################################")
    print("Results:")
    for k, v in metrics.items():
        print(k + ":", v)
    print("#####################################")
    return 0.0


test_eval_hook = eval_hooks[FLAGS.model](reader, dataset, epoch_interval=1, side_effect=side_effect)
test_eval_hook.at_test_time(1)

print("Done!")
