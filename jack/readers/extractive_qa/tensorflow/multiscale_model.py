"""
This file contains FastQA specific modules and ports
"""

from jack.core import *
from jack.readers.extractive_qa.shared import XQAPorts, get_answer_and_span
from jack.readers.extractive_qa.tensorflow.abstract_model import AbstractXQAModelModule
from jack.readers.extractive_qa.tensorflow.answer_layer import compute_question_state, compute_spans
from jack.tfutil import misc
from jack.tfutil import sequence_encoder
from jack.tfutil.embedding import conv_char_embedding
from jack.tfutil.highway import highway_network
from jack.tfutil.segment import segment_softmax


class MultiscalePorts:
    predicted_span_scores = TensorPort(tf.float32, [None], 'predicted_span_scores')
    segmentation_indicator = TensorPort(tf.float32, [None, None, None], 'segmentation_indicator')
    layer_logits = TensorPort(tf.float32, [None, None, None], 'layer_logits')


class MultiscaleQA(AbstractXQAModelModule):
    _input_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                    XQAPorts.emb_support, XQAPorts.support_length, XQAPorts.support2question,
                    # char embedding inputs
                    XQAPorts.word_chars, XQAPorts.word_char_length,
                    XQAPorts.question_words, XQAPorts.support_words,
                    # feature input
                    XQAPorts.word_in_question,
                    # optional input, provided only during training
                    XQAPorts.answer2support_training,
                    XQAPorts.is_eval]

    @property
    def input_ports(self) -> Sequence[TensorPort]:
        return self._input_ports

    _output_ports = [XQAPorts.start_scores, XQAPorts.end_scores, XQAPorts.span_prediction,
                     MultiscalePorts.segmentation_indicator]

    @property
    def output_ports(self) -> Sequence[TensorPort]:
        return self._output_ports

    def create_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        with tf.variable_scope("fast_qa", initializer=tf.contrib.layers.xavier_initializer()):
            # Some helpers
            batch_size = tf.shape(tensors.question_length)[0]
            max_question_length = tf.reduce_max(tensors.question_length)
            support_mask = misc.mask_for_lengths(tensors.support_length)

            input_size = shared_resources.config["repr_dim_input"]
            size = shared_resources.config["repr_dim"]
            with_char_embeddings = shared_resources.config.get("with_char_embeddings", False)

            # set shapes for inputs
            tensors.emb_question.set_shape([None, None, input_size])
            tensors.emb_support.set_shape([None, None, input_size])

            emb_question = tensors.emb_question
            emb_support = tensors.emb_support
            if with_char_embeddings:
                # compute combined embeddings
                [char_emb_question, char_emb_support] = conv_char_embedding(
                    len(shared_resources.char_vocab), size, tensors.word_chars, tensors.word_char_length,
                    [tensors.question_words, tensors.support_words])

                emb_question = tf.concat([emb_question, char_emb_question], 2)
                emb_support = tf.concat([emb_support, char_emb_support], 2)
                input_size += size

                # set shapes for inputs
                emb_question.set_shape([None, None, input_size])
                emb_support.set_shape([None, None, input_size])

            # compute encoder features
            question_features = tf.ones(tf.stack([batch_size, max_question_length, 2]))

            v_wiqw = tf.get_variable("v_wiq_w", [1, 1, input_size],
                                     initializer=tf.constant_initializer(1.0))

            wiq_w = tf.matmul(tf.gather(emb_question * v_wiqw, tensors.support2question), emb_support, adjoint_b=True)
            wiq_w = wiq_w + tf.expand_dims(support_mask, 1)

            question_binary_mask = tf.gather(tf.sequence_mask(tensors.question_length, dtype=tf.float32),
                                             tensors.support2question)
            wiq_w = tf.reduce_sum(tf.nn.softmax(wiq_w) * tf.expand_dims(question_binary_mask, 2), [1])

            # [B, L , 2]
            support_features = tf.stack([tensors.word_in_question, wiq_w], 2)

            # highway layer to allow for interaction between concatenated embeddings
            if with_char_embeddings:
                with tf.variable_scope("char_embeddings") as vs:
                    emb_question = tf.layers.dense(emb_question, size, name="embeddings_projection")
                    emb_question = highway_network(emb_question, 1)
                    vs.reuse_variables()
                    emb_support = tf.layers.dense(emb_support, size, name="embeddings_projection")
                    emb_support = highway_network(emb_support, 1)

            keep_prob = 1.0 - shared_resources.config.get("dropout", 0.0)
            emb_question, emb_support = tf.cond(
                tensors.is_eval,
                lambda: (emb_question, emb_support),
                lambda: (tf.nn.dropout(emb_question, keep_prob, noise_shape=[1, 1, emb_question.get_shape()[-1].value]),
                         tf.nn.dropout(emb_support, keep_prob, noise_shape=[1, 1, emb_question.get_shape()[-1].value]))
            )

            # extend embeddings with features
            emb_question_ext = tf.concat([emb_question, question_features], 2)
            emb_support_ext = tf.concat([emb_support, support_features], 2)

            # encode question and support
            encoded_question = sequence_encoder.bi_rnn(
                size, tf.contrib.rnn.GRUBlockCell(size), emb_question_ext, tensors.question_length,
                with_projection=True)
            num_layers = shared_resources.config["num_layers"]
            encoded_support_fw, encoded_support_bw, segm_indicator = sequence_encoder.multi_scale_birnn_encoder(
                size, shared_resources.config["num_layers"],
                tf.contrib.rnn.GRUBlockCell(size), emb_support_ext, tensors.support_length)

            start_scores, end_scores, doc_idx, predicted_start_pointer, predicted_end_pointer = \
                multiscale_answer_layer(size, num_layers, encoded_question, tensors.question_length,
                                        encoded_support_fw, encoded_support_bw, tensors.support_length,
                                        segm_indicator,
                                        tensors.support2question, tensors.answer2support, tensors.is_eval,
                                        beam_size=shared_resources.config.get("beam_size", 1),
                                        max_span_size=shared_resources.config.get("max_span_size", 10000))

            span = tf.stack([doc_idx, predicted_start_pointer, predicted_end_pointer], 1)

            return TensorPort.to_mapping(self.output_ports, (start_scores, end_scores, span, segm_indicator))

        #    _training_input_ports = [MultiscalePorts.layer_logits, MultiscalePorts.segmentation_indicator,
        #                             XQAPorts.answer_span, XQAPorts.answer2support_training, XQAPorts.support2question]

        #    @property
        #    def training_input_ports(self) -> Sequence[TensorPort]:
        #        return self._training_input_ports

    def create_training_output_old(self, shared_resources, training_input_tensors):
        num_layers = shared_resources.config['num_layers']
        tensors = TensorPortTensors(training_input_tensors)
        num_questions = tf.reduce_max(tensors.support2question) + 1

        # [B, L, num_layers]
        segm_indicator = tensors.segmentation_indicator
        logits = tensors.layer_logits

        # only consider positions where a segment is created
        mask = (segm_indicator - 1.0) * 1e6
        logits += mask

        # [B, L * num_layers]
        probs = segment_softmax(tf.reshape(logits, [tf.shape(logits)[0], -1]), tensors.support2question)
        # [B, L, num_layers]
        probs = tf.reshape(probs, tf.shape(logits))

        # [B, L, num_layers]
        # for each token (and layer) we compute its containing segment index
        _, segm_index = segmentation_indicator_to_span(tf.to_int32(segm_indicator), 1)

        # [A]
        start, end = tensors.answer_span[:, 0], tensors.answer_span[:, 1]

        # [A, num_layers]
        correct_segm_starts = tf.gather_nd(segm_index, tf.stack([tensors.answer2support, start], 1))
        correct_segm_ends = tf.gather_nd(segm_index, tf.stack([tensors.answer2support, end], 1))

        # [A * num_layers]
        correct_segm_starts = tf.reshape(correct_segm_starts, [-1])
        correct_segm_ends = tf.reshape(correct_segm_ends, [-1])

        # [A, L, num_layers]
        probs = tf.gather(probs, tensors.answer2support)
        # [A, num_layers, L]
        probs = tf.transpose(probs, [0, 2, 1])
        # [A * num_layers, L]
        probs = tf.reshape(probs, [-1, tf.shape(probs)[2]])
        num_answer_and_layers = tf.shape(probs)[0]
        correct_answer_prob_start = tf.gather_nd(
            probs, tf.stack([tf.range(num_answer_and_layers), correct_segm_starts], 1))
        correct_answer_prob_end = tf.gather_nd(
            probs, tf.stack([tf.range(num_answer_and_layers), correct_segm_ends], 1))

        correct_answer_prob_start = tf.reshape(correct_answer_prob_start, [-1, num_layers])
        correct_answer_prob_end = tf.reshape(correct_answer_prob_end, [-1, num_layers])

        best_layer_for_answer = tf.argmax(correct_answer_prob_start * correct_answer_prob_end, axis=1,
                                          output_type=tf.int32)

        correct_answer_prob_start = tf.reduce_max(correct_answer_prob_start, axis=1)
        correct_answer_prob_end = tf.reduce_max(correct_answer_prob_end, axis=1)

        answer2question = tf.gather(tensors.support2question, tensors.answer2support)
        if shared_resources.config.get('loss', 'sum') == 'sum':
            span_probs = tf.unsorted_segment_sum(
                correct_answer_prob_start * correct_answer_prob_end, answer2question, num_questions)
        else:
            span_probs = tf.unsorted_segment_max(
                correct_answer_prob_start * correct_answer_prob_end, answer2question, num_questions)
        loss = -tf.reduce_mean(tf.log(span_probs + 1e-6))

        ###### supervise segmentation ######
        # [A]
        segm_indicator = segm_indicator
        end_segment = tf.gather_nd(segm_indicator, tf.stack([tensors.answer2support, end, best_layer_for_answer], 1))
        start_segment = tf.gather_nd(segm_indicator,
                                     tf.stack([tensors.answer2support, tf.maximum(start - 1, 0), best_layer_for_answer],
                                              1))
        start_segment = tf.where(start > 0, start_segment, tf.ones_like(start_segment))
        segm_border_loss = 2.0 - end_segment - start_segment

        left_mask = 1.0 - tf.sequence_mask(start, tf.shape(segm_indicator)[1], tf.float32)
        right_mask = tf.sequence_mask(end, tf.shape(segm_indicator)[1], tf.float32)
        segm_intra_mask = tf.expand_dims(left_mask * right_mask, 2)

        # [A, num_layers]
        segm_intra_loss = tf.reduce_sum(tf.gather(segm_indicator, tensors.answer2support) * segm_intra_mask, 1)

        # [A]
        segm_intra_loss = tf.gather_nd(segm_intra_loss, tf.stack(
            [tf.range(tf.shape(tensors.answer2support)[0]), best_layer_for_answer], 1))
        # segm_intra_mask_sum = tf.reduce_sum(segm_intra_mask, 1)
        segm_intra_loss /= tf.maximum(tf.reduce_sum(segm_intra_mask, 1), 1.0)

        # segm_mask = tf.minimum(1.0, tf.to_float(end - start))

        # r = tf.expand_dims(tf.range(1, num_layers, dtype=tf.float32), 0)
        segm_loss = segm_border_loss + 2.0 * segm_intra_loss
        segm_loss = tf.reduce_mean(segm_loss)

        tf.summary.scalar("segmentation_loss", segm_loss)
        tf.summary.scalar("loss", loss)

        loss = tf.Print(loss, [loss, segm_loss, best_layer_for_answer], summarize=10)

        return {Ports.loss: loss + 10.0 * segm_loss}


class XQAMultiScaleOutputModule(OutputModule):
    def __init__(self, shared_resources):
        self.beam_size = shared_resources.config.get("beam_size", 1)

    def __call__(self, questions, span_prediction,
                 token_offsets, selected_support, support2question,
                 span_scores):
        all_answers = []
        for k, q in enumerate(questions):
            answers = []
            doc_idx_map = [i for i, q_id in enumerate(support2question) if q_id == k]
            for j in range(self.beam_size):
                i = k * self.beam_size + j
                doc_idx, start, end = span_prediction[i]
                score = span_scores[i]
                answer, doc_idx, span = get_answer_and_span(
                    q, doc_idx, start, end, token_offsets[doc_idx_map[doc_idx]],
                    [i for q_id, i in zip(support2question, selected_support) if q_id == k])
                answers.append(Answer(answer, span=span, doc_idx=doc_idx, score=score))
            all_answers.append(answers)

        return all_answers

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.answer_span, XQAPorts.token_offsets,
                XQAPorts.selected_support, XQAPorts.support2question, XQAPorts.start_scores]


def segmentation_indicator_to_span(segm_indicator, axis=0):
    new_shape = tf.unstack(tf.shape(segm_indicator))
    new_shape[axis] = 1
    segm_index = segm_indicator * tf.tile(
        tf.reshape(tf.range(1, tf.shape(segm_indicator)[axis] + 1),
                   [1 if i != axis else -1 for i in range(len(new_shape))]), new_shape)
    if axis != 0:
        t = [i for i, _ in enumerate(segm_indicator.get_shape())]
        t[0] = axis
        t[axis] = 0
        segm_index = tf.transpose(segm_index, t)

    ends = tf.scan(lambda acc, x: tf.where(x > 0, x, acc), tf.reverse(segm_index, [0])) - 1
    ends = tf.maximum(ends, 0)
    ends = tf.reverse(ends, [0])
    if axis != 0:
        ends = tf.transpose(ends, t)

    starts = tf.scan(lambda acc, x: tf.where(x > 0, x, acc), segm_index[:-1])
    if axis != 0:
        starts = tf.transpose(starts, t)
    starts = tf.concat([tf.zeros(new_shape[:axis] + [1] + new_shape[axis + 1:], tf.int32), starts], axis)

    return starts, ends


def multiscale_answer_layer(size, num_layers, encoded_question, question_length,
                            encoded_support_fw, encoded_support_bw, support_length, segm_indicator,
                            support2question, answer2support, is_eval, beam_size=1,
                            max_span_size=10000):
    """Answer layer for multiple paragraph QA."""
    # computing single time attention over question
    batch_size = tf.shape(encoded_support_fw)[0]
    # [B, L, num_layers]
    starts, ends = segmentation_indicator_to_span(tf.to_int32(segm_indicator), 1)

    max_length = tf.shape(segm_indicator)[1]
    b_indices = tf.tile(tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis, tf.newaxis], [1, max_length, num_layers])
    l_indices = tf.tile(tf.range(num_layers, dtype=tf.int32)[tf.newaxis, tf.newaxis, :], [batch_size, max_length, 1])

    flat_start = tf.stack([tf.reshape(b_indices, [-1]), tf.reshape(starts, [-1]), tf.reshape(l_indices, [-1])], 1)
    flat_end = tf.stack([tf.reshape(b_indices, [-1]), tf.reshape(ends, [-1]), tf.reshape(l_indices, [-1])], 1)

    # enc_size = encoded_support_fw.get_shape()[-1].value
    encoded_support_fw_aligned = tf.reshape(tf.gather_nd(encoded_support_fw, flat_end), tf.shape(encoded_support_fw))
    encoded_support_bw_aligned = tf.reshape(tf.gather_nd(encoded_support_bw, flat_start), tf.shape(encoded_support_bw))

    encoded_support = tf.concat([encoded_support_fw_aligned, encoded_support_bw_aligned], 3)
    # encoded_support.set_shape([None, 2 * enc_size])
    encoded_support = tf.layers.dense(encoded_support, size, tf.tanh, use_bias=False)
    # encoded_support = tf.reshape(encoded_support, tf.unstack(tf.shape(encoded_support_fw))[:3] + [size])

    question_state = compute_question_state(encoded_question, question_length)

    question_state_trafo = tf.layers.dense(question_state, num_layers * size, use_bias=False)
    question_state_trafo = tf.reshape(question_state_trafo, [tf.shape(question_state_trafo)[0], num_layers, size])
    start_scores = tf.einsum('acd,abcd->abc', tf.gather(question_state_trafo, support2question), encoded_support)
    start_scores = tf.Print(start_scores, [tf.stack([starts[0], ends[0]], 2)], summarize=100)

    question_state_trafo = tf.layers.dense(question_state, num_layers * size, use_bias=False)
    question_state_trafo = tf.reshape(question_state_trafo, [tf.shape(question_state_trafo)[0], num_layers, size])
    end_scores = tf.einsum('acd,abcd->abc', tf.gather(question_state_trafo, support2question), encoded_support)

    # = (tf.stop_gradient(segm_indicator) - 1.0) * 1e6
    support_mask = misc.mask_for_lengths(support_length)
    start_scores += tf.expand_dims(support_mask, 2)  # + mask
    end_scores += tf.expand_dims(support_mask, 2)  # + mask

    return compute_spans(tf.reduce_sum(start_scores, 2), tf.reduce_sum(start_scores, 2), answer2support, is_eval,
                         support2question,
                         beam_size=beam_size,
                         max_span_size=max_span_size)


    # doc_idx, best_indices, best_scores = segment_top_k(tf.reshape(scores, [batch_size, -1]), support2question, beam_size)
    # doc_idx = tf.reshape(doc_idx, [-1])
    # best_indices = tf.reshape(best_indices, [-1])
    # best_scores = tf.reshape(best_scores, [-1])

    # best_layers, best_indices = tf.mod(best_indices, num_layers), tf.div(best_indices, num_layers)

    # best_starts = tf.gather_nd(starts, tf.stack([doc_idx, best_indices, best_layers], 1))
    # best_ends = tf.gather_nd(ends, tf.stack([doc_idx, best_indices, best_layers], 1))

    # compute correct doc idx per question
    # _, _, num_doc_per_question = tf.unique_with_counts(support2question)
    # offsets = tf.cumsum(num_doc_per_question, exclusive=True)
    # doc_idx_for_support = tf.range(tf.shape(support2question)[0]) - tf.gather(offsets, support2question)

    # return scores, tf.gather(doc_idx_for_support, doc_idx), best_starts, best_ends, best_scores
