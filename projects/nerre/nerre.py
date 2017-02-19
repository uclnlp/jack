import os
from collections import defaultdict
from typing import NamedTuple, Sequence, Mapping

import tensorflow as tf
import numpy as np
import copy

# tensorflow inputs
from jtr.preprocess.vocab import Vocab
#from jtr.preprocess.batch import get_feed_dicts
from jtr.preprocess.batch import numpify, get_batches, GeneratorWithRestart
from jtr.util import tfutil
from projects.nerre.data import read_ann, convert_to_batchable_format, convert_batch_to_ann
from projects.nerre.eval import calculateMeasures


tokens = tf.placeholder(tf.int32, [None, None], name="sentences_as_ints")  # [batch_size, max_num_tokens]
sentence_lengths = tf.placeholder(tf.int32, [None], name="sentence_length")
tag_labels = tf.placeholder(tf.int32, [None, None], name="bio_labels_as_ints")  # [batch_size, max_num_tokens]
type_labels = tf.placeholder(tf.int32, [None, None], name="type_labels_as_ints")  # [batch_size, max_num_tokens]
target_relations = tf.placeholder(tf.int32, [None, None, None], "relation_matrices")  # [batch_size, max_num_tokens, max_num_tokens]]
document_indices = tf.placeholder(tf.int32, [None], name="document_indices") # [batch_size]
token_char_offsets = tf.placeholder(tf.int32, [None, None, None], name="token_char_offsets")

placeholders = {"sentences_as_ints": tokens, "sentence_length": sentence_lengths, "bio_labels_as_ints": tag_labels,
                "type_labels_as_ints": type_labels, "relation_matrices": target_relations,
                "document_indices": document_indices, "token_char_offsets": token_char_offsets}


def create_model(output_size, layers, dropout, num_words, emb_dim, max_sent_len, tag_size=3, type_size=4, rel_size=3):
    with tf.variable_scope("embeddings"):
        embeddings = tf.get_variable("embeddings", shape=[num_words, emb_dim], dtype=tf.float32)

    with tf.variable_scope("input"):
        embedded_input = tf.gather(embeddings, tokens)  # embed sentence tokens

    with tf.variable_scope("model"):
        cell = tf.nn.rnn_cell.LSTMCell(
            output_size,
            state_is_tuple=True,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        if layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers)

        if dropout != 0.0:
            cell_dropout = \
                tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0 - dropout)
        else:
            cell_dropout = cell

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_dropout,
            cell_dropout,
            embedded_input,
            sequence_length=sentence_lengths,
            dtype=tf.float32
        )

        # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )  [batch_size, max_time, cell_fw.output_size] for fw and bw each
        output = tf.concat(2, outputs)  # concatenate along output_size dimension -> [batch_size, max_time, cell_fw.output_size*2]

        dim1, dim2, dim3 = tf.unpack(tf.shape(target_relations))

        # masking or sentence lengths  -- doesn't seem to help
        #output_mask = mask_for_lengths(sentence_lengths, dim1, max_length=max_sent_len, dim2=emb_dim*2, value=-1000)
        #output = output + output_mask

        output_with_tags = tf.contrib.layers.linear(output, tag_size)

        output_with_tags_softm = tf.nn.softmax(output_with_tags)
        predicted_tags = tf.arg_max(output_with_tags_softm, 2)
        loss_tags = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output_with_tags, tag_labels))

        output_with_labels = tf.contrib.layers.linear(output, type_size)
        output_with_labels_softm = tf.nn.softmax(output_with_labels)
        predicted_labels = tf.arg_max(output_with_labels_softm, 2)
        loss_labels = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output_with_labels, type_labels))

        output_with_rels = tf.contrib.layers.linear(output, max_sent_len * rel_size)
        output_with_rels_reshaped = tf.reshape(output_with_rels, [dim1, dim2, dim3, -1])
        output_with_labels_softm = tf.nn.softmax(output_with_rels_reshaped)
        predicted_rels = tf.arg_max(output_with_labels_softm, 3)
        predicted_rels = tf.reshape(predicted_rels, [dim1, dim2, dim3])
        loss_rels = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output_with_rels_reshaped, target_relations))

        loss_all = loss_tags + loss_labels + loss_rels
        
        return loss_all, predicted_tags, predicted_labels, predicted_rels
    
    
def train(train_batches, vocab, max_sent_len, max_epochs=200, l2=0.0, learning_rate=0.001, emb_dim=10, output_size=10, layers=1, dropout=0.0, sess=None, clip=None, clip_op=tf.clip_by_value):
    
    loss, predicted_tags, predicted_labels, predicted_rels = create_model(max_sent_len=max_sent_len, output_size=10, layers=1, dropout=0.0, num_words=len(vocab), emb_dim=10)

    optim = tf.train.AdamOptimizer(learning_rate)
    
    if l2 != 0.0:
        loss = loss + tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2

    if clip is not None:
        gradients = optim.compute_gradients(loss)
        if clip_op == tf.clip_by_value:
            capped_gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                                for grad, var in gradients]
        elif clip_op == tf.clip_by_norm:
            capped_gradients = [(tf.clip_by_norm(grad, clip), var)
                                for grad, var in gradients]
        min_op = optim.apply_gradients(capped_gradients)
    else:
        min_op = optim.minimize(loss)
        
        
    # Do not take up all the GPU memory, all the time.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    if sess is None:
        sess = tf.Session(config=sess_config)

    tf.global_variables_initializer().run(session=sess)

    train_pred_batches = []
    
    for i in range(1, max_epochs + 1):
        #print('iteration', str(i))
        loss_all = []
        for j, batch in enumerate(train_batches):
            _, current_loss = sess.run([min_op, loss], feed_dict=batch)
            loss_all.append(current_loss)

            # in last epoch, get predictions on training data
            if i == max_epochs:
                train_pred_batches_i = {}

                curr_predicted_tags, curr_predicted_labels, curr_predicted_rels \
                    = sess.run([predicted_tags, predicted_labels, predicted_rels], feed_dict=batch)

                train_pred_batches_i["bio_labels_as_ints"], train_pred_batches_i["type_labels_as_ints"], train_pred_batches_i["relation_matrices"] \
                    = curr_predicted_tags, curr_predicted_labels, curr_predicted_rels

                train_pred_batches_i["sentences_as_ints"], train_pred_batches_i["sentence_length"], train_pred_batches_i["document_indices"], train_pred_batches_i["token_char_offsets"]\
                    = batch[placeholders["sentences_as_ints"]], batch[placeholders["sentence_length"]], batch[placeholders["document_indices"]], batch[placeholders["token_char_offsets"]]

                train_pred_batches.append(train_pred_batches_i)

        print("Epoch:", i, "\tAverage loss:", np.average(loss_all))

    return train_pred_batches



def get_feed_dicts(data, placeholders, batch_size=32, pad=0, bucket_order=None, bucket_structure=None,
                   exact_epoch=False):
    """Creates feed dicts for all batches with a given batch size.

    Args:
        `data` (dict): The input data for the feed dicts.
        `placeholders` (dict): The TensorFlow placeholders for the data
            (placeholders.keys() must form a subset of data.keys()).
        `batch_size` (int): The batch size for the data.
        `pad` (int): Padding symbol index to pad lists of different sizes.
        `bucket_order`: argument `order` in get_buckets (list with keys); `None` if no bucketing
        `bucket_structure`: argument `structure` in get_buckets; `None` if no bucketing
        `exact_epoch`: if set to `True`, final batch per bucket may be smaller, but each instance will be seen exactly
            once during training. Default: `False`, to be certain during training
            that each instance per batch gets same weight in the total loss.

    Returns:
        GeneratorWithRestart: Generator that yields a feed_dict for each
        iteration. A feed dict consists of '{ placeholder : data-batch }` key-value pairs.
    """
    assert isinstance(data, dict) and isinstance(placeholders, dict)
    assert set(placeholders.keys()).issubset(set(data.keys())), \
        'data keys %s \nnot compatible with placeholder keys %s' % (set(placeholders.keys()), set(data.keys()))

    def generator():
        batches = get_batches(data, batch_size, pad, bucket_order, bucket_structure, exact_epoch)
        # fixme: this is potentially inefficient as it might be called every time we retrieve a batch
        # todo: measure and fix if significant impact
        mapped = map(lambda xs: {placeholders[k]: xs[k] for k in placeholders}, batches)
        # for each key in placeholders dict, pair the placeholder with the corresponding batch dict value
        for x in mapped:
            yield x

    return GeneratorWithRestart(generator)


def mask_for_lengths(lengths, batch_size=None, max_length=None, dim2=None, mask_right=True,
                         value=-1000.0):

    """
    Creates a [batch_size x max_length] mask.
    :param lengths: int32 1-dim tensor of batch_size lengths
    :param batch_size: int32 0-dim tensor or python int
    :param max_length: int32 0-dim tensor or python int
    :param mask_right: if True, everything before "lengths" becomes zero and the
        rest "value", else vice versa
    :param value: value for the mask
    :return: [batch_size x max_length] mask of zeros and "value"s
    """
    if max_length is None:
        max_length = tf.reduce_max(lengths)
    if batch_size is None:
        batch_size = tf.shape(lengths)[0]
    # [batch_size x max_length x dim2]
    mask = tf.reshape(tf.tile(tf.range(0, max_length), [batch_size * dim2]), tf.pack([batch_size, max_length, dim2]))
    if mask_right:
        mask = tf.greater_equal(mask, tf.expand_dims(tf.expand_dims(lengths, 1), 1))
    else:
        mask = tf.less(mask, tf.expand_dims(tf.expand_dims(lengths, 1), 1))
    mask = tf.cast(mask, tf.float32) * value
    return mask


if __name__ == "__main__":
    train_dir = "/Users/Isabelle/Documents/UCLMR/semeval2017-orga/data/train2"
    dev_dir = "/Users/Isabelle/Documents/UCLMR/semeval2017-orga/data/dev/"
    
    #train_instances = read_ann(train_dir)
    dev_instances = read_ann(dev_dir)
    
    print("Loaded {} dev instances".format(len(dev_instances)))
    
    vocab = Vocab()

    numinst = 10
    for instance in dev_instances: #+ dev_instances:
        #numinst -= 1
        #if numinst == 0:
        #    break
        for sent in instance.doc:
            for token in sent.tokens:
                vocab(token.word)
    
    print("Collected {} word types".format(len(vocab)))

    train_batchable = convert_to_batchable_format(dev_instances, vocab)

    batch_size = 16
    data_np = numpify(train_batchable, pad=0)
    max_sent_len = len(data_np["relation_matrices"][0][0])

    train_feed_dicts = get_feed_dicts(data_np, placeholders, batch_size,
                       bucket_order=None, bucket_structure=None)

    train_preds = train(train_feed_dicts, vocab, max_sent_len, max_epochs=200, emb_dim=50, output_size=50)

    for batch in train_preds:
        convert_batch_to_ann(batch, dev_instances, "/tmp")

    calculateMeasures(dev_dir, "/tmp/")