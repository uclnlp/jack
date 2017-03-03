import os
from collections import defaultdict
from typing import NamedTuple, Sequence, Mapping

import tensorflow as tf
import numpy as np
import copy
import sys

# tensorflow inputs
from jtr.preprocess.vocab import Vocab
#from jtr.preprocess.batch import get_feed_dicts
from jtr.preprocess.batch import numpify, get_batches, GeneratorWithRestart
from jtr.util import tfutil
from projects.nerre.data import read_ann, convert_to_batchable_format, convert_batch_to_ann, reset_output_dir
from projects.nerre.eval import calculateMeasures


def create_placeholders():
    tokens = tf.placeholder(tf.int32, [None, None], name="sentences_as_ints")  # [batch_size, max_num_tokens]
    sentence_lengths = tf.placeholder(tf.int32, [None], name="sentence_length")
    tag_labels = tf.placeholder(tf.int32, [None, None], name="bio_labels_as_ints")  # [batch_size, max_num_tokens]
    type_labels = tf.placeholder(tf.int32, [None, None], name="type_labels_as_ints")  # [batch_size, max_num_tokens]
    target_relations = tf.placeholder(tf.int32, [None, None, None],
                                      "relation_matrices")  # [batch_size, max_num_tokens, max_num_tokens]]
    document_indices = tf.placeholder(tf.int32, [None], name="document_indices")  # [batch_size]
    token_char_offsets = tf.placeholder(tf.int32, [None, None, None], name="token_char_offsets")

    placeholders = {"sentences_as_ints": tokens, "sentence_length": sentence_lengths, "bio_labels_as_ints": tag_labels,
                "type_labels_as_ints": type_labels, "relation_matrices": target_relations,
                "document_indices": document_indices, "token_char_offsets": token_char_offsets}

    return placeholders


def create_model(placeholders, output_size, layers, dropout, num_words, emb_dim, max_sent_len, tag_size=3, type_size=4, rel_size=3, relations=True, tieOutputLayer=True, hierarchicalSoftm=True, a = 1, b = 1, c = 1):
    # sequence-to-sequence bi-LSTM with hierarchical softmax for keyphrase identification, classification and relation classification
    with tf.variable_scope("embeddings"):
        embeddings = tf.get_variable("embeddings", shape=[num_words, emb_dim], dtype=tf.float32)

    with tf.variable_scope("input"):
        embedded_input = tf.gather(embeddings, placeholders["sentences_as_ints"])  # embed sentence tokens

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
            sequence_length=placeholders["sentence_length"],
            dtype=tf.float32
        )

        # states = (states_fw, states_bw) = ( (c_fw, h_fw), (c_bw, h_bw) )  [batch_size, max_time, cell_fw.output_size] for fw and bw each
        output = tf.concat(2, outputs)  # concatenate along output_size dimension -> [batch_size, max_time, cell_fw.output_size*2]

        dim1, dim2, dim3 = tf.unpack(tf.shape(placeholders["relation_matrices"]))

        # masking output for sentence lengths  -- doesn't seem to help
        #output_mask = mask_for_lengths(sentence_lengths, dim1, max_length=max_sent_len, dim2=emb_dim*2, value=-1000)
        #output = output + output_mask

        output_mask_keys = mask_for_lengths(placeholders["sentence_length"], dim1, max_length=max_sent_len, value=0, mask_right=False)
        output_mask_rels = mask_for_lengths(placeholders["sentence_length"], dim1, max_length=max_sent_len, dim2=dim3, value=0, mask_right=False)

        output_with_tags = tf.contrib.layers.linear(output, tag_size)

        output_with_tags_softm = tf.nn.softmax(output_with_tags)
        predicted_tags = tf.arg_max(output_with_tags_softm, 2)
        l_tags = tf.nn.sparse_softmax_cross_entropy_with_logits(output_with_tags, placeholders["bio_labels_as_ints"])

        # applying sentence length mask
        #l_tags = l_tags + output_mask_keys
        l_tags = tf.select(output_mask_keys, l_tags, tf.zeros(tf.shape(l_tags)))

        loss_tags = tf.reduce_sum(l_tags)

        output_with_labels = tf.contrib.layers.linear(output, type_size)

        # set output_with_labels [batch_size, max_num_tokens, 0] to output_with_tags [batch_size, max_num_tokens, 0]  -- i.e. the params for O should be the same for tags and labels
        if tieOutputLayer == True:
            output_with_tags_slice = tf.slice(output_with_tags, [0, 0, 0], [dim1, dim2, 1])
            output_with_labels_slice = tf.slice(output_with_labels, [0, 0, 1], [dim1, dim2, type_size-1])
            output_with_labels = tf.concat(values=[output_with_tags_slice, output_with_labels_slice], concat_dim=2)

        output_with_labels_softm = tf.nn.softmax(output_with_labels)

        predicted_labels = tf.arg_max(output_with_labels_softm, 2)
        l_labels = tf.nn.sparse_softmax_cross_entropy_with_logits(output_with_labels, placeholders["type_labels_as_ints"])

        # applying sentence length mask
        l_labels = tf.select(output_mask_keys, l_labels, tf.zeros(tf.shape(l_labels)))

        loss_labels = tf.reduce_sum(l_labels)

        if relations == True:
            output_with_rels = tf.contrib.layers.linear(output, max_sent_len * rel_size)
            output_with_rels = tf.reshape(output_with_rels, [dim1, dim2, dim3, -1])  # batch_size, seq_len, seq_len, num_rels

            # the params for [O, O] relation weights should be same as those for O tags and labels
            # do this separately for the rows and columns in the rel label matrix
            if tieOutputLayer == True:
                output_rels_unpacked1 = tf.unpack(output_with_rels, num=max_sent_len, axis=1)  # produces dim2 number of [dim1, dim3, dim4] slices
                #output_with_rels = tf.pack(output_rels_unpacked1, axis=1)

                outputs1 = tf.TensorArray(size=max_sent_len, dtype='float32', infer_shape=False)
                i = 0
                for rel_sl in output_rels_unpacked1:  # this is of len max_sent_len
                    output_with_rels_slice = tf.slice(rel_sl, [0, 0, 1], [dim1, dim3, rel_size - 1])  # these are all the non-O weights # bio_vocab("B")
                    output_with_labels_slice = tf.slice(output_with_tags, [0, 0, 0], [dim1, dim2, 1])  # these are the O weights

                    # multiply the weights for the B labels of keyphrase identification with the weights for both relations in the rel matrix
                    if hierarchicalSoftm == True:
                        output_with_labels_slice_B = tf.slice(rel_sl, [0, 0, 1], [dim1, dim3, 2])
                        output_with_rels_slice = tf.einsum('blr,blk->blr', output_with_rels_slice, output_with_labels_slice_B)

                    output_with_rels = tf.concat(values=[output_with_labels_slice, output_with_rels_slice], concat_dim=2)
                    outputs1 = outputs1.write(i, output_with_rels)
                    i += 1

                # packs along axis 0, there doesn't seem to be a way to change that (?)
                output_with_rels = outputs1.pack()
                output_with_rels = tf.transpose(output_with_rels, perm=[1, 0, 2, 3])


                output_rels_unpacked2 = tf.unpack(output_with_rels, num=max_sent_len, axis=2)  # produces dim1 number of [dim1, dim2, dim4] slices

                outputs2 = tf.TensorArray(size=max_sent_len, dtype='float32', infer_shape=False)
                i = 0
                for rel_sl in output_rels_unpacked2:  # this is of len max_sent_len
                    output_with_rels_slice = tf.slice(rel_sl, [0, 0, 1], [dim1, dim3, rel_size - 1])  # these are all the non-O weights
                    output_with_labels_slice = tf.slice(output_with_tags, [0, 0, 0], [dim1, dim2, 1])  # these are the O weights

                    # multiply the weights for the B labels of keyphrase identification with the weights for both relations in the rel matrix
                    if hierarchicalSoftm == True:
                        output_with_labels_slice_B = tf.slice(rel_sl, [0, 0, 1], [dim1, dim3, 2])
                        output_with_rels_slice = tf.einsum('blr,blk->blr', output_with_rels_slice,
                                                           output_with_labels_slice_B)

                    output_with_rels = tf.concat(values=[output_with_labels_slice, output_with_rels_slice], concat_dim=2)
                    outputs2 = outputs2.write(i, output_with_rels)
                    i += 1

                # packs along axis 0, there doesn't seem to be a way to change that (?)
                output_with_rels = outputs2.pack()
                output_with_rels = tf.transpose(output_with_rels, perm=[1, 0, 2, 3])


            output_with_labels_softm = tf.nn.softmax(output_with_rels)
            predicted_rels = tf.arg_max(output_with_labels_softm, 3)
            predicted_rels = tf.reshape(predicted_rels, [dim1, dim2, dim3])

            l_relations = tf.nn.sparse_softmax_cross_entropy_with_logits(output_with_rels, placeholders["relation_matrices"])

            # applying sentence length mask
            l_relations = tf.select(output_mask_rels, l_relations, tf.zeros(tf.shape(l_relations)))

            loss_rels = tf.reduce_sum(l_relations)

            loss_all = a * loss_tags + b * loss_labels + c * loss_rels

        else:
            predicted_rels = placeholders["relation_matrices"]
            loss_all = loss_tags + loss_labels
        
        return loss_all, predicted_tags, predicted_labels, predicted_rels

    
    
def train(placeholders, train_batches, dev_batches, vocab, max_sent_len, max_epochs=200, l2=0.0, learning_rate=0.001, emb_dim=10, output_size=10, layers=1, dropout=0.0, sess=None, clip=None, clip_op=tf.clip_by_value, pred_on_train=False, a=1, b=1, c=1, useGoldKeyphr=False):
    
    loss, predicted_tags, predicted_labels, predicted_rels = create_model(placeholders, max_sent_len=max_sent_len, output_size=output_size, layers=layers, dropout=dropout, num_words=len(vocab), emb_dim=emb_dim, a=a, b=b, c=c)

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

    pred_batches = []

    i = 0
    j = 1
    last_loss = sys.maxsize
    fiveagoloss = sys.maxsize
    while (i < max_epochs-1 or last_loss > 1.0):
        loss_all = []
        for j, batch in enumerate(train_batches):
            _, current_loss = sess.run([min_op, loss], feed_dict=batch)
            loss_all.append(current_loss)

            # in last epoch, get predictions on training data
            if i == max_epochs and pred_on_train == True:
                train_pred_batches_i = {}

                curr_predicted_tags, curr_predicted_labels, curr_predicted_rels \
                    = sess.run([predicted_tags, predicted_labels, predicted_rels], feed_dict=batch)

                train_pred_batches_i["bio_labels_as_ints"], train_pred_batches_i["type_labels_as_ints"], train_pred_batches_i["relation_matrices"] \
                    = curr_predicted_tags, curr_predicted_labels, curr_predicted_rels

                train_pred_batches_i["sentences_as_ints"], train_pred_batches_i["sentence_length"], train_pred_batches_i["document_indices"], train_pred_batches_i["token_char_offsets"]\
                    = batch[placeholders["sentences_as_ints"]], batch[placeholders["sentence_length"]], batch[placeholders["document_indices"]], batch[placeholders["token_char_offsets"]]

                pred_batches.append(train_pred_batches_i)

        average_loss = float(np.average(loss_all))
        print("Epoch:", i, "\tAverage loss:", average_loss)

        # early stopping
        if (average_loss > last_loss and average_loss > fiveagoloss):
            print("Stopping early", str(average_loss), str(last_loss), str(fiveagoloss))
            break

        last_loss = average_loss
        if j ==5:
            fiveagoloss = average_loss
            j = 0

        if pred_on_train == False:
            for j, dev_batch in enumerate(dev_batches):
                dev_pred_batches_i = {}

                curr_predicted_tags, curr_predicted_labels, curr_predicted_rels \
                    = sess.run([predicted_tags, predicted_labels, predicted_rels], feed_dict=dev_batch)

                if useGoldKeyphr == False:
                    dev_pred_batches_i["bio_labels_as_ints"], dev_pred_batches_i["type_labels_as_ints"] = curr_predicted_tags, curr_predicted_labels
                    dev_pred_batches_i["relation_matrices"] = dev_batch[placeholders["relation_matrices"]]
                else:
                    dev_pred_batches_i["bio_labels_as_ints"], dev_pred_batches_i["type_labels_as_ints"], \
                    dev_pred_batches_i["relation_matrices"] \
                        = dev_batch[placeholders["bio_labels_as_ints"]], dev_batch[placeholders["type_labels_as_ints"]], \
                          dev_batch[placeholders["relation_matrices"]]

                dev_pred_batches_i["sentences_as_ints"], dev_pred_batches_i["sentence_length"], \
                dev_pred_batches_i["document_indices"], dev_pred_batches_i["token_char_offsets"] \
                    = dev_batch[placeholders["sentences_as_ints"]], dev_batch[placeholders["sentence_length"]], dev_batch[
                    placeholders["document_indices"]], dev_batch[placeholders["token_char_offsets"]]

                pred_batches.append(dev_pred_batches_i)

        i += 1
        j -=1

    return pred_batches



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

    if dim2 != None:
        # [batch_size x max_length x dim2]
        mask = tf.reshape(tf.tile(tf.range(0, max_length), [batch_size * dim2]), tf.pack([batch_size, max_length, dim2]))
        if mask_right:
            mask = tf.greater_equal(mask, tf.expand_dims(tf.expand_dims(lengths, 1), 1))
        else:
            mask = tf.less(mask, tf.expand_dims(tf.expand_dims(lengths, 1), 1))
    else:
        # [batch_size x max_length]
        mask = tf.reshape(tf.tile(tf.range(0, max_length), [batch_size]), tf.pack([batch_size, -1]))
        if mask_right:
            mask = tf.greater_equal(mask, tf.expand_dims(lengths, 1))
        else:
            mask = tf.less(mask, tf.expand_dims(lengths, 1))
    # otherwise we return a boolean mask
    if value != 0:
        mask = tf.cast(mask, tf.float32) * value
    return mask


if __name__ == "__main__":
    train_dir = "/Users/Isabelle/Documents/UCLMR/semeval2017-orga/data/train2"
    dev_dir = "/Users/Isabelle/Documents/UCLMR/semeval2017-orga/data/dev/"
    
    train_instances = read_ann(train_dir)
    dev_instances = read_ann(dev_dir)
    
    print("Loaded {} dev instances".format(len(dev_instances)))
    
    vocab = Vocab()

    numinst = 10
    for instance in dev_instances:
        for sent in instance.doc:
            for token in sent.tokens:
                vocab(token.word)
    
    print("Collected {} word types".format(len(vocab)))

    train_batchable = convert_to_batchable_format(train_instances, vocab)
    dev_batchable = convert_to_batchable_format(dev_instances, vocab)

    batch_size = 1

    data_all = {}
    for key, value in train_batchable.items():
        data_all[key] = train_batchable[key] + dev_batchable[key]

    data_np = numpify(data_all, pad=0)
    data_train_np = {}
    data_dev_np = {}
    for key, value in data_np.items():
        data_train_np[key] = data_np[key][0:len(train_batchable[key])]
        data_dev_np[key] = data_np[key][len(train_batchable[key]):len(train_batchable[key])+len(dev_batchable[key])]

    max_sent_len = len(data_train_np["relation_matrices"][0][0])


    for dim in [300, 200, 100]:
        for drop in [0.2, 0.5]:
            for l2 in [0.2, 0.5]:
                for a in [1, 2, 3]:
                    for b in [1, 2, 3]:
                        print("Training with dim", dim, "drop", drop, "l2", l2, "a", a, "b", b)

                        placeholders = create_placeholders()

                        train_feed_dicts = get_feed_dicts(data_train_np, placeholders, batch_size,
                                                          bucket_order=None, bucket_structure=None)

                        dev_feed_dicts = get_feed_dicts(data_dev_np, placeholders, batch_size,
                                                    bucket_order=None, bucket_structure=None)

                        dev_preds = train(placeholders, train_feed_dicts, dev_feed_dicts, vocab, max_sent_len, max_epochs=500, emb_dim=dim, output_size=dim, l2=l2, dropout=drop, a=a, b=b, c=1, useGoldKeyphr=False)

                        reset_output_dir()

                        for dev_batch in dev_preds:
                            convert_batch_to_ann(dev_batch, dev_instances, out_dir=out_dir)

                        calculateMeasures(dev_dir, "/tmp/", ignoremissing=False)

                        tf.reset_default_graph()