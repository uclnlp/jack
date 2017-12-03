from typing import Sequence

import tensorflow as tf

from jack.core import Ports, TensorPort, TensorPortTensors
from jack.core.tensorflow import TFModelModule
from jack.readers.extractive_qa.shared import XQAPorts
from jack.tfutil import sequence_encoder
from jack.tfutil.xqa import xqa_crossentropy_loss


class AbstractXQAModelModule(TFModelModule):
    _input_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                    XQAPorts.emb_support, XQAPorts.support_length,
                    XQAPorts.support2question,
                    # char embedding inputs
                    XQAPorts.word_chars, XQAPorts.word_char_length,
                    XQAPorts.question_words, XQAPorts.support_words,
                    # optional input, provided only during training
                    XQAPorts.answer2support_training, XQAPorts.is_eval]
    _output_ports = [XQAPorts.start_scores, XQAPorts.end_scores,
                     XQAPorts.span_prediction]
    _training_input_ports = [XQAPorts.start_scores, XQAPorts.end_scores,
                             XQAPorts.answer_span, XQAPorts.answer2support_training, XQAPorts.support2question]
    _training_output_ports = [Ports.loss]

    @property
    def output_ports(self) -> Sequence[TensorPort]:
        return self._output_ports

    @property
    def input_ports(self) -> Sequence[TensorPort]:
        return self._input_ports

    @property
    def training_input_ports(self) -> Sequence[TensorPort]:
        return self._training_input_ports

    @property
    def training_output_ports(self) -> Sequence[TensorPort]:
        return self._training_output_ports

    @staticmethod
    def rnn_encoder(size, sequence, seq_length, encoder_type='lstm', reuse=False, with_projection=False,
                    name='encoder'):
        if encoder_type == 'lstm':
            return sequence_encoder.bi_lstm(size, sequence, seq_length, name, reuse, with_projection)
        elif encoder_type == 'sru':
            with_residual = sequence.get_shape()[2].value == size
            return sequence_encoder.bi_sru(size, sequence, seq_length, with_residual, name, reuse, with_projection)
        elif encoder_type == 'gru':
            return sequence_encoder.bi_rnn(size, tf.contrib.rnn.BlockGRUCell(size), sequence,
                                           seq_length, name, reuse, with_projection)
        else:
            raise ValueError("Unknown encoder type: %s" % encoder_type)

    @staticmethod
    def conv_encoder(size, sequence, num_layers=3, width=3,
                     dilations=[1, 2, 4, 8, 16, 1], encoder_type='gldr', reuse=False, name='encoder'):
        if encoder_type == 'gldr':
            return sequence_encoder.gated_linear_dilated_residual_network(size, sequence, dilations, width, name, reuse)
        elif encoder_type == 'convnet':
            return sequence_encoder.gated_linear_convnet(size, sequence, num_layers, width, name, reuse)
        else:
            raise ValueError("Unknown encoder type: %s" % encoder_type)

    def create_training_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        return {
            Ports.loss: xqa_crossentropy_loss(tensors.start_scores, tensors.end_scores,
                                              tensors.answer_span, tensors.answer2support,
                                              tensors.support2question,
                                              use_sum=shared_resources.config.get('loss', 'sum') == 'sum')
        }
