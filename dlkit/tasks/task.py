import hjson
from typing import Dict, Union, Callable, List
import torch
import torch.nn as nn
from torch.nn import functional as F
from dlkit.models import MODEL_REGISTRY, MODEL_CONFIG_REGISTRY
from dlkit.utils.config import ConfigTool
from . import task_config_register, task_register

import pytorch_lightning as pl

@task_config_register('basic')
class TaskConfig(object):
    """docstring for TaskConfig"""
    def __init__(self, config):
        super(TaskConfig, self).__init__()
        self.model_config = config.get("model", {})
        self.optimizer_config = config.pop("optimizer", {})
        self.loss_config = config.pop('loss', {})

@task_register("basic")
class Task(pl.LightningModule):
    """
    """
    def __init__(self, task_config: TaskConfig, paras: Config):
        super().__init__()
        self.task_config = task_config
        model_config = task_config.get("model")
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
        result = self.model.training_step(batch)
        loss = F.cross_entropy(result.get(''), batch.get("label"))
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = self.init_module(self.task_config.optimizer_config, MODEL_REGISTRY, MODEL_CONFIG_REGISTRY)
        return optimizer
