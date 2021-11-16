import hjson
from typing import Dict, Union, Callable, List
import torch
import torch.nn as nn
from torch.nn import functional as F
from dlkit.models import model_register, model_config_register
from dlkit.optimizers import optimizer_register, optimizer_config_register
from dlkit.losses import loss_register, loss_config_register
from dlkit.utils.config import ConfigTool
from . import task_config_register, task_register

import pytorch_lightning as pl

@task_config_register('basic')
class BasicTaskConfig(object):
    """docstring for TaskConfig"""
    def __init__(self, config):
        super(BasicTaskConfig, self).__init__()
        self.model, self.model_config = self.get_model(config.get("model"))
        self.loss, self.loss_config = self.get_loss(config.get("loss"))
        self.optimizer, self.optimizer_config = self.get_optimizer(config.get("optimizer"))

        self.config = config.pop('config', {})

    def get_model(self, config):
        """get embedding config and embedding module

        :config: TODO
        :returns: TODO

        """
        return ConfigTool.get_leaf_module(model_register, model_config_register, "model", config)
        
    def get_loss(self, config):
        """get encoder config and encoder module

        :config: TODO
        :returns: TODO

        """
        return ConfigTool.get_leaf_module(loss_register, loss_config_register, "loss", config)
        
    def get_optimizer(self, config):
        """get decoder config and decoder module

        :config: TODO
        :returns: TODO
        """
        return ConfigTool.get_leaf_module(optimizer_register, optimizer_config_register, "optimizer", config)


@task_register("basic")
class BasicTask(pl.LightningModule):
    """
    """
    def __init__(self, config: BasicTaskConfig):
        super().__init__()
        self.model = config.model(config.model_config)
        self.calc_loss = config.loss(config.loss_config)
        self.get_optimizer = config.optimizer(model=self.model, config=config.optimizer_config)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        result = self.model.training_step(batch)
        loss = self.calc_loss(result)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer, schedual = self.get_optimizer()
        return optimizer, schedual
