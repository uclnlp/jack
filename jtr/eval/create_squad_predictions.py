import json

import tensorflow as tf

from jtr.convert.SQuAD2jtr import convert_squad
from jtr.jack.data_structures import convert2qasettings
from jtr.jack.readers import readers
from jtr.load.embeddings import load_embeddings
from jtr.preprocess.vocab import Vocab

tf.app.flags.DEFINE_string('file', None, 'dataset file')
tf.app.flags.DEFINE_string('dataset_type', 'squad', 'either squad or jtr')
tf.app.flags.DEFINE_string('model', None, 'Name of the reader')
tf.app.flags.DEFINE_string('model_dir', None, 'directory to saved model')
tf.app.flags.DEFINE_string('embedding_path', None, 'path to embeddings')
tf.app.flags.DEFINE_string('embedding_format', 'glove', 'embeddings format')
tf.app.flags.DEFINE_string('device', "/cpu:0", 'device to use')
tf.app.flags.DEFINE_string('out', "results.json", 'Result file path.')
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

reader.setup_from_file(FLAGS.model_dir)

if FLAGS.dataset_type == "squad":
    dataset_jtr = convert_squad(FLAGS.file)
else:
    with open(FLAGS.file) as f:
        dataset_jtr = json.load(f)

dataset = convert2qasettings(dataset_jtr)

print("Start!")
answers = reader.process_outputs(dataset, FLAGS.batch_size, debug=True)
results = {dataset[i][0].id: a.text for i, a in enumerate(answers)}
with open(FLAGS.out, "w") as out_file:
    json.dump(results, out_file)

print("Done!")
