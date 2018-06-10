from typing import Sequence

from jack.core import Ports, TensorPort, TensorPortTensors
from jack.core.tensorflow import TFModelModule
from jack.readers.extractive_qa.shared import XQAPorts
from jack.util.tf.xqa import xqa_crossentropy_loss


class AbstractXQAModelModule(TFModelModule):
    _input_ports = [XQAPorts.emb_question, XQAPorts.question_length,
                    XQAPorts.emb_support, XQAPorts.support_length, XQAPorts.support2question,
                    # char embedding inputs
                    XQAPorts.word_chars, XQAPorts.word_char_length,
                    XQAPorts.question_batch_words, XQAPorts.support_batch_words,
                    # feature input
                    XQAPorts.word_in_question,
                    # optional input, provided only during training
                    XQAPorts.correct_start, XQAPorts.answer2support_training,
                    XQAPorts.is_eval]

    _output_ports = [XQAPorts.start_scores, XQAPorts.end_scores,
                     XQAPorts.answer_span]
    _training_input_ports = [XQAPorts.start_scores, XQAPorts.end_scores,
                             XQAPorts.answer_span_target, XQAPorts.answer2support_training, XQAPorts.support2question]
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

    def create_training_output(self, shared_resources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        return {
            Ports.loss: xqa_crossentropy_loss(tensors.start_scores, tensors.end_scores,
                                              tensors.answer_span_target, tensors.answer2support,
                                              tensors.support2question,
                                              use_sum=shared_resources.config.get('loss', 'sum') == 'sum')
        }
