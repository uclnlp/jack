import logging
import sys
from abc import abstractmethod
from typing import Mapping, List, Iterable, Tuple

import numpy as np
import torch
from torch import nn

from jack.core import ModelModule, SharedResources, TensorPort
from jack.core import reader
from jack.core.data_structures import Answer
from jack.core.data_structures import QASetting
from jack.core.tensorport import Ports

logger = reader.logger


class PyTorchModelModule(ModelModule):
    """This class represents an  abstract ModelModule for PyTorch models.

    It requires the implementation of 2 nn.modules that create predictions and the training outputs for the defined
    the ports.
    """

    def __init__(self, shared_resources: SharedResources):
        self.shared_resources = shared_resources
        # will be set in setup later
        self._prediction_module = None
        self._loss_module = None

    def __call__(self, batch: Mapping[TensorPort, np.ndarray],
                 goal_ports: List[TensorPort] = None) -> Mapping[TensorPort, np.ndarray]:
        """Runs a batch and returns values/outputs for specified goal ports.
        Args:
            batch: mapping from ports to values
            goal_ports: optional output ports, defaults to output_ports of this module will be returned

        Returns:
            A mapping from goal ports to tensors.
        """
        goal_ports = goal_ports or self.output_ports
        inputs = [p.create_torch_variable(batch.get(p), gpu=torch.cuda.device_count() > 0) for p in self.input_ports]
        outputs = self.prediction_module.forward(*inputs)
        ret = {p: p.torch_to_numpy(t) for p, t in zip(self.output_ports, outputs) if p in goal_ports}
        for p in goal_ports:
            if p not in ret and p in batch:
                ret[p] = batch[p]
        return ret

    @abstractmethod
    def create_prediction_module(self, shared_resources: SharedResources) -> nn.Module:
        """Creates and returns a PyTorch nn.Module for computing predictions.

        It takes inputs as defined by `input_ports` and produces  outputs as defined by `output_ports`"""
        raise NotImplementedError

    @abstractmethod
    def create_loss_module(self, shared_resources: SharedResources) -> nn.Module:
        """Creates and returns a PyTorch nn.Module for computing output necessary for training, such as a loss.

        It takes inputs as defined by `training_input_ports` and produces outputs as defined by
        `training_output_ports`."""
        raise NotImplementedError

    @property
    def prediction_module(self) -> nn.Module:
        return self._prediction_module

    @property
    def loss_module(self) -> nn.Module:
        return self._loss_module

    def setup(self, is_training=True):
        """Sets up the module.

        This usually involves creating the actual tensorflow graph. It is expected to be called after the input module
        is set up and shared resources, such as the vocab, config, etc., are prepared already at this point.
        """
        self._prediction_module = self.create_prediction_module(self.shared_resources)
        self._loss_module = self.create_loss_module(self.shared_resources)
        if torch.cuda.device_count() > 0:
            self._prediction_module.cuda()
            self._loss_module.cuda()

    def store(self, path):
        with open(path, 'wb') as f:
            torch.save({'prediction_module': self.prediction_module.state_dict(),
                        'loss_module': self.loss_module.state_dict()}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            d = torch.load(f)
        self.prediction_module.load_state_dict(d['prediction_module'])
        self.loss_module.load_state_dict(d['loss_module'])


class PyTorchReader(reader.JTReader):
    """Tensorflow implementation of JTReader.

    A tensorflow reader reads inputs consisting of questions, supports and possibly candidates, and produces answers.
    It consists of three layers: input to tensor (input_module), tensor to tensor (model_module), and tensor to answer
    (output_model). These layers are called in-turn on a given input (list).
    """

    @property
    def model_module(self) -> PyTorchModelModule:
        return super().model_module

    def train(self, optimizer,
              training_set: Iterable[Tuple[QASetting, List[Answer]]],
              batch_size: int, max_epochs=10, hooks=tuple(), **kwargs):
        """This method trains the reader (and changes its state).

        Args:
            optimizer: optimizer
            training_set: the training instances.
            batch_size: size of training batches
            max_epochs: maximum number of epochs
            hooks: TrainingHook implementations that are called after epochs and batches
        """
        logger.info("Setting up data and model...")
        if not self._is_setup:
            # First setup shared resources, e.g., vocabulary. This depends on the input module.
            self.setup_from_data(training_set, is_training=True)
        batches = self.input_module.batch_generator(training_set, batch_size, is_eval=False)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        loss_idx = self.model_module.training_output_ports.index(Ports.loss)

        logger.info("Start training...")
        p_module = self.model_module.prediction_module
        l_module = self.model_module.loss_module
        for i in range(1, max_epochs + 1):
            for j, batch in enumerate(batches):
                for p, v in batch.items():
                    if isinstance(p, TensorPort):
                        batch[p] = p.create_torch_variable(v, gpu=torch.cuda.device_count() > 0)

                # zero the parameter gradients
                optimizer.zero_grad()
                pred_outputs = p_module.forward(
                    *(batch[p] for p in self.model_module.input_ports))
                batch.update(zip(self.model_module.output_ports, pred_outputs))
                train_outputs = l_module.forward(
                    *(batch[p] for p in self.model_module.training_input_ports))
                current_loss = train_outputs[loss_idx]
                current_loss.backward()
                optimizer.step()

                for hook in hooks:
                    hook.at_iteration_end(i, current_loss.data[0], set_name='train')

            # calling post-epoch hooks
            for hook in hooks:
                hook.at_epoch_end(i)
