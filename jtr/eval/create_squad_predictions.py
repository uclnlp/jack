import json
import math
import sys
import tensorflow as tf

from jtr.convert.SQuAD2jtr import convert_squad
from jtr.jack.readers import readers
from jtr.preprocess.vocab import Vocab
from jtr.load.embeddings import load_embeddings
from jtr.jack.data_structures import convert2qasettings

tf.app.flags.DEFINE_string('file', None, 'dataset file')
tf.app.flags.DEFINE_string('reader', None, 'Name of the reader')
tf.app.flags.DEFINE_string('model_dir', None, 'directory to saved model')
tf.app.flags.DEFINE_string('embedding_path', None, 'path to embeddings')
tf.app.flags.DEFINE_string('embedding_format', 'glove', 'embeddings format')
tf.app.flags.DEFINE_string('device', "/cpu:0", 'device to use')
tf.app.flags.DEFINE_string('out', "results.json", 'Result file path.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
# tf.app.flags.DEFINE_integer('beam_size', 1, 'beam size')

FLAGS = tf.app.flags.FLAGS

# vocab
emb = load_embeddings(FLAGS.embedding_path, FLAGS.embedding_format)
vocab = Vocab(emb=emb, init_from_embeddings=True)

reader = readers[FLAGS.reader](vocab, {})
reader.setup_from_file(FLAGS.model_dir)

squad_jtr = convert_squad(FLAGS.file)
squad = convert2qasettings(squad_jtr)

num_batches = math.ceil(len(squad) / FLAGS.batch_size)
results = dict()

counter = 0
for b in range(num_batches):
    i = b * FLAGS.batch_size
    batch = [qa_setting for qa_setting, _ in squad[i: i + FLAGS.batch_size]]
    for i, a in enumerate(reader(batch)):
        results[batch[i].id] = a.text
    counter += len(batch)
    sys.stdout.write("\r%d" % counter)
    sys.stdout.flush()

with open(FLAGS.out, "w") as out_file:
    json.dump(results, out_file)
