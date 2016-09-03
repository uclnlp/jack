from quebap.tensorizer import *
from quebap.util.tfutil import *
import json
from quebap.projects.clozecompose.tensorizer import SequenceTensorizer


def test_cloze_tensorizer(fpath):
    #with open('../quebap/data/snippet/LS/snippet_quebapformat.json') as data_file:
    #with open('../quebap/quebap/data/snippet/SNLI_v1/snippet_quebapformat.json') as data_file:
    #with open('../../../quebap/data/snippet/scienceQA/scienceQA.json') as data_file:
    with open(fpath) as data_file:
        data = json.load(data_file)

    tensorizer = SequenceTensorizer(data)
    feed_dict = next(tensorizer.create_batches(data, batch_size=2))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for placeholder in feed_dict:
            print(placeholder)
            print_tensor_shape_op = tf.Print(placeholder, [tf.shape(placeholder)], "shape: ")
            print(sess.run(print_tensor_shape_op, feed_dict=feed_dict))
            print()

#test_cloze_tensorizer()
