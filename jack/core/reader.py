# -*- coding: utf-8 -*-

"""
Here we define jack readers. Jack readers consist of 3 layers, one that transform
jack data structures into tensors, one that processes predicts the outputs and losses
using a TensorFlow model into other tensors, and one that converts these tensors back to jack data structures.
"""

import logging
import math
import os
import shutil
from typing import Iterable, List

import progressbar

from jack.core.data_structures import *
from jack.core.input_module import InputModule
from jack.core.model_module import ModelModule
from jack.core.output_module import OutputModule
from jack.core.shared_resources import SharedResources

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
        answers = self.output_module(inputs, {p: output_module_input[p] for p in self.output_module.input_ports})
        return answers

    def process_dataset(self, dataset: Sequence[Tuple[QASetting, Answer]], batch_size: int, silent=True):
        """
        Similar to the call method, only that it works on a labeled dataset and applies batching. However, assumes
        that batches in input_module.batch_generator are processed in order and do not get shuffled during with
        flag is_eval set to true.

        Args:
            dataset:
            batch_size: note this information is needed here, but does not set the batch_size the model is using.
            This has to happen during setup/configuration.
            silent: if true, no output

        Returns:
            predicted outputs/answers to a given (labeled) dataset
        """
        batches = self.input_module.batch_generator(dataset, batch_size, is_eval=True)
        answers = list()
        enumerator = enumerate(batches)
        if not silent:
            logger.info("Start answering...")
            bar = progressbar.ProgressBar(
                max_value=math.ceil(len(dataset) / batch_size),
                widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') '])
            enumerator = bar(enumerator)
        for j, batch in enumerator:
            output_module_input = self.model_module(batch, self.output_module.input_ports)
            questions = [q for q, a in dataset[j * batch_size:(j + 1) * batch_size]]
            answers.extend(a[0] for a in self.output_module(
                questions, {p: output_module_input[p] for p in self.output_module.input_ports}))

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
