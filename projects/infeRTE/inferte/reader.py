# -*- coding: utf-8 -*-

import logging
import sys
from typing import Sequence, Mapping, Dict, Any
import tensorflow as tf

from jtr.jack.core import InputModule, ModelModule, OutputModule, Ports

logger = logging.getLogger(__name__)


class JTReader:
    """
    A Reader reads inputs consisting of questions, supports and possibly candidates, and produces answers.
    It consists of three layers: input to tensor (input_module), tensor to tensor (model_module), and tensor to answer
    (output_model). These layers are called in-turn on a given input (list).
    """

    def __init__(self,
                 config: dict,
                 input_module: InputModule,
                 model_module: ModelModule,
                 output_module: OutputModule,
                 sess: tf.Session = None):
        self.config = config
        self.sess = sess
        self.output_module = output_module
        self.model_module = model_module
        self.input_module = input_module

        if self.sess is None:
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=sess_config)

        assert all(port in self.input_module.output_ports for port in self.model_module.input_ports), \
            "Input Module outputs must include model module inputs"

        assert all(port in self.input_module.training_ports or port in self.model_module.output_ports or
                   port in self.input_module.output_ports for port in self.model_module.training_input_ports), \
            "Input Module (training) outputs and model module outputs must include model module training inputs"

        assert all(port in self.model_module.output_ports or port in self.input_module.output_ports
                   for port in self.output_module.input_ports), \
            "Module model output must match output module inputs"

    def __call__(self, inputs: Sequence[Mapping]) -> Sequence[dict]:
        """
        Answers a list of question settings
        Args:
            inputs: a list of inputs.

        Returns:
            predicted outputs/answers to a given (labeled) dataset
        """
        batch = self.input_module(inputs)
        output_module_input = self.model_module(self.sess, batch, self.output_module.input_ports)
        answers = self.output_module(inputs, *[output_module_input[p] for p in self.output_module.input_ports])
        return answers

    def process_outputs(self, dataset: Dict[str, Any],
                        batch_size: int,
                        debug=False):
        """
        Similar to the call method, only that it works on a labeled dataset and applies batching. However, assumes
        that batches in input_module.dataset_generator are processed in order and do not get shuffled during with
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
        batches = self.input_module.dataset_generator(dataset)
        answers = list()
        logger.debug("Start answering...")
        for j, batch in enumerate(batches):
            output_module_input = self.model_module(self.sess, batch, self.output_module.input_ports)
            inputs = [x for x, _ in dataset[j*batch_size:(j+1)*batch_size]]
            answers.extend(
                self.output_module(inputs, *[output_module_input[p] for p in self.output_module.input_ports]))
            if debug:
                sys.stdout.write("\r%d/%d examples processed..." % (len(answers), len(dataset)))
                sys.stdout.flush()
        return answers

    def train(self,
              optimizer,
              training_set: Dict[str, Any],
              max_epochs=10, hooks=[],
              l2=0.0, clip=None, clip_op=tf.clip_by_value):
        """
        This method trains the reader (and changes its state).
        Args:
            training_set: the training instances.
            max_epochs: maximum number of epochs
            hooks: TrainingHook implementations that are called after epochs and batches
            l2: whether to use l2 regularization
            clip: whether to apply gradient clipping and at which value
            clip_op: operation to perform for clipping
        """
        self.setup()

        batches = self.input_module.dataset_generator(training_set)
        loss = self.model_module.tensors[Ports.loss]

        if l2:
            loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2

        if clip:
            gradients = optimizer.compute_gradients(loss)
            if clip_op == tf.clip_by_value:
                gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var) for grad, var in gradients]
            elif clip_op == tf.clip_by_norm:
                gradients = [(tf.clip_by_norm(grad, clip), var) for grad, var in gradients]
            min_op = optimizer.apply_gradients(gradients)
        else:
            min_op = optimizer.minimize(loss)

        # initialize non model variables like learning rate, optim vars ...
        self.sess.run([v.initializer for v in tf.global_variables() if v not in self.model_module.variables])

        logger.info("Start training...")
        for i in range(1, max_epochs + 1):
            for j, batch in enumerate(batches):
                feed_dict = self.model_module.convert_to_feed_dict(batch)
                _, current_loss = self.sess.run([min_op, loss], feed_dict=feed_dict)

                for hook in hooks:
                    hook.at_iteration_end(i, current_loss)

            # calling post-epoch hooks
            for hook in hooks:
                hook.at_epoch_end(i)

    def setup(self):
        """
        Sets up modules given a training dataset if necessary.
        Args:
            data: training dataset
        """
        self.input_module.setup()
        self.model_module.setup(is_training=True)
        self.output_module.setup()

        self.sess.run([v.initializer for v in self.model_module.variables])
