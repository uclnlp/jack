# -*- coding: utf-8 -*-

import logging
from abc import *
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Mapping

import numpy as np
import tensorflow as tf

from jtr.jack import JTReader, TensorPort, Answer, Question, FlatPorts

logger = logging.getLogger(__name__)


# todo: hooks should also have prefixes so that one can use the same hook with different parameters
class TrainingHook(metaclass=ABCMeta):
    """Serves as Hook interface."""
    @abstractproperty
    def reader(self) -> JTReader:
        """ Returns: JTReader instance"""

    @abstractmethod
    def at_epoch_end(self, epoch: int, **kwargs):
        pass

    @abstractmethod
    def at_iteration_end(self, epoch: int, loss: float, **kwargs):
        pass


class TraceHook(TrainingHook):
    """Abstract hook class, which implements an update function the summary."""
    def __init__(self, reader, summary_writer=None):
        self._summary_writer = summary_writer
        self._reader = reader

    @property
    def reader(self) -> JTReader:
        return self._reader

    def update_summary(self, sess, current_step, title, value):
        """Adds summary (title, value) to summary writer object.

        Args:
            sess (TensorFlow session): The TensorFlow session object.
            current_step (int): Current step in the training procedure.
            value (float): Scalar value for the message.
        """
        if self._summary_writer is not None:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag=title, simple_value=value),
            ])
            self._summary_writer.add_summary(summary, current_step)


class LossHook(TraceHook):
    """A hook at prints the current loss and adds it to the summary."""
    def __init__(self, reader, iter_interval=None, summary_writer=None):
        super(LossHook, self).__init__(reader, summary_writer)
        self._iter_interval = iter_interval
        self._acc_loss = 0
        self._iter = 0
        self._epoch_loss = 0
        self._iter_epoch = 0

    def at_iteration_end(self, epoch, loss, **kwargs):
        """Prints the loss, epoch, and #calls; adds it to the summary. Loss should be batch normalized."""
        self._iter_epoch += 1
        self._epoch_loss += 1
        if self._iter_interval is None:
            return loss

        self._iter += 1
        self._acc_loss += loss
        if not self._iter == 0 and self._iter % self._iter_interval == 0:
            loss = self._acc_loss / self._iter_interval
            logger.info("Epoch {}\tIteration {}\tLoss {}".format(str(epoch), str(self._iter), str(loss)))
            self.update_summary(self.reader.sess, self._iter, "Loss", loss)
            self._acc_loss = 0

        return self._acc_loss / self._iter

    def at_epoch_end(self, epoch, **kwargs):
        if self._iter_interval is None:
            loss = self._acc_loss / self._iter_interval
            logger.info("Epoch {}\tIteration {}\tLoss {}".format(str(epoch), str(self._iter), str(loss)))
            self.update_summary(self.reader.sess, self._iter, "Loss", loss)
            self._epoch_loss = 0
            self._iter_epoch = 0

        self._iter = 0
        return self._epoch_loss / self._iter_epoch


class EvalHook(TraceHook):
    def __init__(self, reader: JTReader, dataset: List[Tuple[Question, List[Answer]]], ports: List[TensorPort],
                 iter_interval=None, epoch_interval=1, metrics=None, summary_writer=None,
                 write_metrics_to=None, info="", side_effect=None):
        super(EvalHook, self).__init__(reader, summary_writer)
        self._dataset = dataset
        self._batches = None
        self._total = len(dataset)
        self._ports = ports
        self._epoch_interval = epoch_interval
        self._iter_interval = iter_interval
        #self.done_for_epoch = False
        self._iter = 0
        self._info = info or self.__class__.__name__
        self._write_metrics_to = write_metrics_to
        self._metrics = metrics or self.possible_metrics
        self._side_effect = side_effect
        self._side_effect_state = None

    @abstractproperty
    def possible_metrics(self) -> List[str]:
        """Returns: list of metric keys this evaluation hook produces. """

    @abstractmethod
    def apply_metrics(self, tensors: Mapping[TensorPort, np.ndarray]) -> Mapping[str, float]:
        """Returns: dict from metric name to float"""

    def combine_metrics(self, accumulated_metrics: Mapping[str, List[float]]) -> Mapping[str, float]:
        """Returns:
               dict from metric name to float. Per default batch metrics are simply averaged by
               total number of examples"""
        return {k: sum(vs)/self._total for k, vs in accumulated_metrics.items()}

    def __call__(self, epoch):
        logger.info("Started evaluation %s" % self._info)

        if self._batches is None:
            self._batches = self.reader.input_module.dataset_generator(self._dataset, is_eval=True)

        metrics = defaultdict(lambda: list())
        for i, batch in enumerate(self._batches):
            predictions = self.reader.model_module(self.reader.sess, batch, self._ports)
            m = self.apply_metrics(predictions)
            for k in self._metrics:
                metrics[k].append(m[k])

        metrics = self.combine_metrics(metrics)

        printmetrics = sorted(metrics.keys())
        res = "Epoch %d\tIteration %d\ttotal %d" % (epoch, self._iter, self._total)
        for m in printmetrics:
            res += '\t%s: %.3f' % (m, metrics[m])
            self.update_summary(self.reader.sess, self._iter, self._info + '_' + m, metrics[m])
            if self._write_metrics_to is not None:
                with open(self._write_metrics_to, 'a') as f:
                    f.write("{0} {1} {2:.5}\n".format(datetime.now(), self._info + '_' + m,
                                                      np.round(metrics[m],5)))
        res += '\t' + self._info
        logger.info(res)

        if self._side_effect is not None:
            self._side_effect_state = self._side_effect(metrics, self._side_effect_state)

    def at_epoch_end(self, epoch: int, **kwargs):
        if self._epoch_interval is not None and epoch % self._epoch_interval == 0:
            self.__call__(epoch)
        self._iter = 0

    def at_iteration_end(self, epoch: int, loss: float, **kwargs):
        self._iter += 1
        if self._iter_interval is not None and self._iter % self._iter_interval == 0:
            self.__call__(epoch)


class XQAEvalHook(EvalHook):
    """This evaluation hook computes the following metrics: exact and per-answer f1 on token basis."""

    def __init__(self, reader: JTReader, dataset: List[Tuple[Question, List[Answer]]],
                 iter_interval=None, epoch_interval=1, metrics=None, summary_writer=None,
                 write_metrics_to=None, info="", side_effect=None, **kwargs):
        ports = [FlatPorts.Prediction.answer_span, FlatPorts.Target.answer_span, FlatPorts.Input.answer_to_question]
        super().__init__(reader, dataset, ports, iter_interval, epoch_interval, metrics, summary_writer,
                         write_metrics_to, info, side_effect)

    @property
    def possible_metrics(self) -> List[str]:
        return ["exact", "f1"]

    def apply_metrics(self, tensors: Mapping[TensorPort, np.ndarray]) -> Mapping[str, float]:
        correct_spans = tensors[FlatPorts.Target.answer_span]
        predicted_spans = tensors[FlatPorts.Prediction.answer_span]
        correct2prediction = tensors[FlatPorts.Input.answer_to_question]

        def len_np_or_list(v):
            if isinstance(v, list):
                return len(v)
            else:
                return v.shape[0]

        acc_f1 = 0.0
        acc_exact = 0.0
        k = 0
        for i in range(len_np_or_list(predicted_spans)):
            f1, exact = 0.0, 0.0
            p_start, p_end = predicted_spans[i][0], predicted_spans[i][1]
            while k < len_np_or_list(correct_spans) and correct2prediction[k] == i:
                c_start, c_end = correct_spans[k][0], correct_spans[k][1]
                if p_start == c_start and p_end == c_end:
                    f1 = 1.0
                    exact = 1.0
                elif f1 < 1.0:
                    total = float(c_end - c_start + 1)
                    missed_from_start = float(p_start - c_start)
                    missed_from_end = float(c_end - p_end)
                    tp = total - min(total, max(0, missed_from_start) + max(0, missed_from_end))
                    fp = max(0, -missed_from_start) + max(0, -missed_from_end)
                    recall = tp / total
                    precision = tp / (tp + fp + 1e-10)
                    f1 = max(f1, 2.0 * precision * recall / (precision + recall + 1e-10))
                k += 1

            acc_f1 += f1
            acc_exact += exact

        return {"f1": acc_f1, "exact": acc_exact}





