import hjson
from typing import Dict, Union, Callable, List
import torch
import torch.nn as nn
from torch.nn import functional as F
from dlkit.models import model_register, model_config_register
from dlkit.optimizers import optimizer_register, optimizer_config_register
from dlkit.losses import loss_register, loss_config_register
from dlkit.utils.config import ConfigTool
from . import imodel_config_register, imodel_register, GatherOutputMixin

import pytorch_lightning as pl

@imodel_config_register('basic')
class BasicIModelConfig(object):
    """docstring for IModelConfig"""
    def __init__(self, config):
        super(BasicIModelConfig, self).__init__()
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


@imodel_register("basic")
class BasicIModel(pl.LightningModule, GatherOutputMixin):
    """
    """
    def __init__(self, config: BasicIModelConfig):
        super().__init__()
        self.model = config.model(config.model_config)
        self.calc_loss = config.loss(config.loss_config)
        self.get_optimizer = config.optimizer(model=self.model, config=config.optimizer_config)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        result = self.model.training_step(batch)
        loss = self.calc_loss(result, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        result = self.model.validation_step(batch)
        loss = self.calc_loss(result, batch)
        return {"loss": loss, "index": batch["_index"], "predict": result}

    def validation_epoch_end(self, outputs):
        """TODO: Docstring for test_epoch_end.
        :returns: TODO

        """
        outputs = self.gather_outputs(outputs)

        if self.local_rank in [0, -1]:
            key_all_ins_map = self.concat_list_of_dict_outputs(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        result = self.model.test_step(batch)
        loss = self.calc_loss(result, batch)
        return {"loss": loss, "index": batch["_index"], "predict": result}

    def test_epoch_end(self, outputs):
        """TODO: Docstring for test_epoch_end.
        :returns: TODO

        """
        outputs = self.gather_outputs(outputs)

        if self.local_rank in [0, -1]:
            key_all_ins_map = self.concat_list_of_dict_outputs(outputs)
        return outputs

    def predict_step(self, batch, batch_idx):
        return self.model.predict_step(batch)

    def configure_optimizers(self):
        optimizer, schedual = self.get_optimizer()
        return optimizer, schedual
