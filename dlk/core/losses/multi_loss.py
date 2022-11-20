# Copyright cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List
import torch.nn as nn
from . import loss_register, loss_config_register
from dlk import additional_function
import torch.nn as nn
import torch
from dlk.utils.config import ConfigTool

@additional_function("multi_loss", "sum")
def loss_sum(losses: Dict[str, torch.Tensor], **args:Dict):
    """sum all losses

    Args:
        losses (List): list of loss

    Returns: 
        sum of losses
    """
    loss = sum([losses[key] for key in losses])
    return loss

@loss_config_register("multi_loss")
class MultiLossConfig(object):
    default_config = {
        "_name": "multi_loss",
        "config": {
            "loss_collect": "sum",
            "args": {},
            "log_map": {
                "loss": "loss"
            },
        },
    }
    """Config for MultiLoss

    Config Example:
    """
    def __init__(self, config: Dict):
        super(MultiLossConfig, self).__init__()
        self.loss_configs = {}
        self.module_rank = []
        self.loss_collect = additional_function.get("multi_loss", config['config']['loss_collect'])
        self.loss_collect_name = config['config']['loss_collect']
        self.log_map = config['config']['log_map']
        if isinstance(self.log_map, str):
            self.log_map = {"loss": self.log_map}
        for loss in config:
            if loss in {'config', "_name", "_base"}:
                continue
            self.module_rank.append(loss)
            module_class, module_config = ConfigTool.get_leaf_module(loss_register, loss_config_register, "loss", config[loss])
            self.loss_configs[loss] = {
                "loss_class": module_class,
                "loss_config": module_config,
            }

@loss_register("multi_loss")
class MultiLoss(nn.Module):
    """ This module is NotImplemented yet don't use it
    """
    def __init__(self, config: MultiLossConfig):
        super(MultiLoss, self).__init__()
        self.config = config
        self.loss_collect = config.loss_collect
        self.module_rank = config.module_rank
        self.losses = nn.ModuleDict({
            loss_name: config.loss_configs[loss_name]['loss_class'](config.loss_configs[loss_name]['loss_config'])
            for loss_name in config.module_rank
        })

    def update_config(self, rt_config):
        """callback for imodel to update the total steps and epochs

        when init the loss module, the total step and epoch is not known, when all data ready, the imodel update the value for loss module

        Args:
            rt_config: { "total_steps": self.num_training_steps, "total_epochs": self.num_training_epochs}

        Returns: 
            None

        """
        for _, loss_module in self.losses.items():
            loss_module.update_config(rt_config)

    def calc(self, result, inputs, rt_config):
        """calc the loss the predict is from result, the ground truth is from inputs

        Args:
            result: the model predict dict
            inputs: the all inputs for model
            rt_config: provide the current training status 
                >>> {
                >>>     "current_step": self.global_step,
                >>>     "current_epoch": self.current_epoch,
                >>>     "total_steps": self.num_training_steps,
                >>>     "total_epochs": self.num_training_epochs
                >>> }

        Returns: 
            loss

        """
        losses = {}
        log_loss = {}
        for loss_module in self.module_rank:
            loss, log = self.losses[loss_module](result, inputs, rt_config)
            losses[loss_module] = loss
            log_loss.update(log)
        loss = self.loss_collect(losses=losses, rt_config=rt_config)
        if self.config.loss_collect_name != 'sum':
            log_loss.update({self.config.log_map.get("sum_loss", "sum_loss"): sum([losses[key] for key in losses])})
        log_loss.update({self.config.log_map['loss']: loss})
        return loss, log_loss

    def __call__(self, result, inputs, rt_config):
        """same as self.calc
        """
        return self.calc(result, inputs, rt_config)
