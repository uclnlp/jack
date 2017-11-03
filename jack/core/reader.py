# -*- coding: utf-8 -*-

"""
Here we define jack readers. Jack readers consist of 3 layers, one that transform
jack data structures into tensors, one that processes predicts the outputs and losses
using a TensorFlow model into other tensors, and one that converts these tensors back to jack data structures.
"""

import logging
import os
import shutil
import sys
from typing import Iterable, List

import tensorflow as tf

from jack.core.data_structures import *
from jack.core.input_module import InputModule
from jack.core.model_module import ModelModule, TFModelModule
from jack.core.output_module import OutputModule
from jack.core.shared_resources import SharedResources
from jack.core.tensorport import Ports

logger = logging.getLogger(__name__)


class JTReader:
    """
    A tensorflow reader reads inputs consisting of questions, supports and possibly candidates, and produces answers.
    It consists of three layers: input to tensor (input_module), tensor to tensor (model_module), and tensor to answer
    (output_model). These layers are called in-turn on a given input (list).
    """

    def __init__(self,
                 shared_resources: SharedResources,
                 input_module: InputModule,
                 model_module: ModelModule,
                 output_module: OutputModule):
        self._shared_resources = shared_resources
        self._output_module = output_module
        self._model_module = model_module
        self._input_module = input_module
        self._is_setup = False

        assert all(port in self.input_module.output_ports for port in self.model_module.input_ports), \
            "Input Module outputs must include model module inputs"

        assert all(port in self.input_module.training_ports or port in self.model_module.output_ports or
                   port in self.input_module.output_ports for port in self.model_module.training_input_ports), \
            "Input Module (training) outputs and model module outputs must include model module training inputs"

        assert all(port in self.model_module.output_ports or port in self.input_module.output_ports
                   for port in self.output_module.input_ports), \
            "Module model output must match output module inputs"

    @property
    def input_module(self) -> InputModule:
        """Returns: input module"""
        return self._input_module

    @property
    def model_module(self) -> ModelModule:
        """Returns: model module"""
        return self._model_module

    @property
    def output_module(self) -> OutputModule:
        """Returns: output module"""
        return self._output_module

    @property
    def shared_resources(self) -> SharedResources:
        """Returns: SharedResources object"""
        return self._shared_resources

    def __call__(self, inputs: Sequence[QASetting]) -> Sequence[Answer]:
        """
        Answers a list of question settings
        Args:
            inputs: a list of inputs.

        Returns:
            predicted outputs/answers to a given (labeled) dataset
        """
        batch = self.input_module(inputs)
        output_module_input = self.model_module(batch, self.output_module.input_ports)
        answers = self.output_module(inputs, *[output_module_input[p] for p in self.output_module.input_ports])
        return answers

    def process_dataset(self, dataset: Sequence[Tuple[QASetting, Answer]], batch_size: int, debug=False):
        """
        Similar to the call method, only that it works on a labeled dataset and applies batching. However, assumes
        that batches in input_module.batch_generator are processed in order and do not get shuffled during with
        flag is_eval set to true.

        Args:
            dataset:
            batch_size: note this information is needed here, but does not set the batch_size the model is using.
            This has to happen during setup/configuration.
            debug: if true, logging counter

        Returns:
            predicted outputs/answers to a given (labeled) dataset
        """
        logger.debug("Setting up batches...")
        batches = self.input_module.batch_generator(dataset, batch_size, is_eval=True)
        answers = list()
        logger.debug("Start answering...")
        for j, batch in enumerate(batches):
            output_module_input = self.model_module(batch, self.output_module.input_ports)
            answers.extend(self.output_module(
                output_module_input, *[output_module_input[p] for p in self.output_module.input_ports]))
            if debug:
                logger.debug("{}/{} examples processed".format(len(answers), len(dataset)))
        return answers

    def train(self, optimizer, training_set: Iterable[Tuple[QASetting, List[Answer]]], batch_size: int,
              max_epochs=10, hooks=tuple(), **kwargs):
        """
        This method trains the reader (and changes its state).

        Args:
            optimizer: TF optimizer
            training_set: the training instances.
            max_epochs: maximum number of epochs
            hooks: TrainingHook implementations that are called after epochs and batches
            kwargs: additional reader specific options
        """
        raise NotImplementedError

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]], is_training=False):
        """
        Sets up modules given a training dataset if necessary.

        Args:
            data: training dataset
            is_training: indicates whether it's the training phase or not
        """
        self.input_module.setup_from_data(data)
        self.input_module.setup()
        self.model_module.setup(is_training)
        self.output_module.setup()
        self._is_setup = True

    def load_and_setup(self, path, is_training=False):
        """
        Sets up already stored reader from model directory.

        Args:
            path: training dataset
            is_training: indicates whether it's the training phase or not
        """
        self.shared_resources.load(os.path.join(path, "shared_resources"))
        self.load_and_setup_modules(path, is_training)

    def load_and_setup_modules(self, path, is_training=False):
        """
        Sets up already stored reader from model directory.

        Args:
            path: training dataset
            is_training: indicates whether it's the training phase or not
        """
        self.input_module.setup()
        self.input_module.load(os.path.join(path, "input_module"))
        self.model_module.setup(is_training)
        self.model_module.load(os.path.join(path, "model_module"))
        self.output_module.setup()
        self.output_module.load(os.path.join(path, "output_module"))
        self._is_setup = True

    def load(self, path):
        """
        (Re)loads module states on a setup reader (but not shared resources).
        If reader is not setup yet use setup from file instead.

        Args:
            path: model directory
        """
        self.input_module.load(os.path.join(path, "input_module"))
        self.model_module.load(os.path.join(path, "model_module"))
        self.output_module.load(os.path.join(path, "output_module"))

    def store(self, path):
        """
        Store module states and shared resources.

        Args:
            path: model directory
        """
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.shared_resources.store(os.path.join(path, "shared_resources"))
        self.input_module.store(os.path.join(path, "input_module"))
        self.model_module.store(os.path.join(path, "model_module"))
        self.output_module.store(os.path.join(path, "output_module"))


class TFReader(JTReader):
    """Tensorflow implementation of JTReader.

    A tensorflow reader reads inputs consisting of questions, supports and possibly candidates, and produces answers.
    It consists of three layers: input to tensor (input_module), tensor to tensor (model_module), and tensor to answer
    (output_model). These layers are called in-turn on a given input (list).
    """

    @property
    def model_module(self) -> TFModelModule:
        return super().model_module

    @property
    def session(self) -> tf.Session:
        """Returns: input module"""
        return self.model_module.tf_session

    def train(self, optimizer,
              training_set: Iterable[Tuple[QASetting, List[Answer]]],
              batch_size: int, max_epochs=10, hooks=tuple(),
              l2=0.0, clip=None, clip_op=tf.clip_by_value, summary_writer=None, **kwargs):
        """
        This method trains the reader (and changes its state).

        Args:
            optimizer: TF optimizer
            training_set: the training instances.
            batch_size: size of training batches
            max_epochs: maximum number of epochs
            hooks: TrainingHook implementations that are called after epochs and batches
            l2: whether to use l2 regularization
            clip: whether to apply gradient clipping and at which value
            clip_op: operation to perform for clipping
        """
        batches, loss, min_op, summaries = self._setup_training(
            batch_size, clip, optimizer, training_set, summary_writer, l2, clip_op, **kwargs)

        self._train_loop(min_op, loss, batches, hooks, max_epochs, summaries, summary_writer, **kwargs)

    def _setup_training(self, batch_size, clip, optimizer, training_set, summary_writer, l2, clip_op, **kwargs):
        logger.info("Setting up data and model...")
        global_step = tf.train.create_global_step()
        if not self._is_setup:
            # First setup shared resources, e.g., vocabulary. This depends on the input module.
            self.setup_from_data(training_set, is_training=True)
        batches = self.input_module.batch_generator(training_set, batch_size, is_eval=False)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        loss = self.model_module.tensors[Ports.loss]
        summaries = None
        if summary_writer is not None:
            summaries = tf.summary.merge_all()
        if l2:
            loss += tf.add_n([tf.nn.l2_loss(v) for v in self.model_module.train_variables]) * l2
        if clip:
            gradients = optimizer.compute_gradients(loss)
            if clip_op == tf.clip_by_value:
                gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                             for grad, var in gradients if grad]
            elif clip_op == tf.clip_by_norm:
                gradients = [(tf.clip_by_norm(grad, clip), var)
                             for grad, var in gradients if grad]
            min_op = optimizer.apply_gradients(gradients, global_step)
        else:
            min_op = optimizer.minimize(loss, global_step)

        # initialize non model variables like learning rate, optimizer vars ...
        self.session.run([v.initializer for v in tf.global_variables() if v not in self.model_module.variables])
        return batches, loss, min_op, summaries

    def _train_loop(self, optimization_op, loss_op, batches, hooks, max_epochs, summaries, summary_writer, **kwargs):
        logger.info("Start training...")
        for i in range(1, max_epochs + 1):
            for j, batch in enumerate(batches):
                feed_dict = self.model_module.convert_to_feed_dict(batch)
                if summaries is not None:
                    step, sums, current_loss, _ = self.session.run(
                        [tf.train.get_global_step(), summaries, loss_op, optimization_op], feed_dict=feed_dict)
                    summary_writer.add_summary(sums, step)
                else:
                    current_loss, _ = self.session.run([loss_op, optimization_op], feed_dict=feed_dict)
                for hook in hooks:
                    hook.at_iteration_end(i, current_loss, set_name='train')

            # calling post-epoch hooks
            for hook in hooks:
                hook.at_epoch_end(i)
