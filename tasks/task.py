import hjson
from typing import Dict, Union, Callable, List
import torch
import torch.nn as nn
from torch.nn import functional as F
from models import MODEL_REGISTRY, MODEL_CONFIG_REGISTRY
from utils.config import Config

import pytorch_lightning as pl

class TaskConfig(Config):
    """docstring for TaskConfig"""
    def __init__(self, **kwargs):
        super(TaskConfig, self).__init__(**kwargs)
        self.model_config = kwargs.pop("model")[0]
        self.optimizer_config = kwargs.pop("optimizer")[0]


class Task(pl.LightningModule):
    """
    """

    def __init__(self, task_config: TaskConfig, paras: Config):
        super().__init__()
        self.task_config = task_config
        self.model = self.init_module(self.task_config.model_config, MODEL_REGISTRY, MODEL_CONFIG_REGISTRY)

    def init_module(self, config: Dict, module_register: Dict, module_config_register: Dict, update_config: Dict={}, module_para: Dict={}):
        """TODO: Docstring for init_module.

        :config: Dict: TODO
        :module_register: TODO
        :module_config_register: TODO
        :returns: TODO

        """
        module_name = config.get('name')
        if not module_name:
            raise KeyError('You must provide a config name.')

        module_config = module_config_register[module_name](config.get('config'), {})
        module = module_register[module_name](module_config)
        return module

    def forward(self, input_dict):
        return self.model(input_dict)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = self.init_module(self.task_config.optimizer_config, MODEL_REGISTRY, MODEL_CONFIG_REGISTRY)
        return optimizer
