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
import torch.nn as nn
import torch
from dlk.utils.config import ConfigTool
from dlk import register, config_register, PROTECTED
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("loss", "multi_loss")
@define
class MultiLossConfig(BaseConfig):
    name = NameField(value="multi_loss", file=__file__, help="multiple loss")
    @define
    class Config:
        loss_collect = StrField(value="sum", checker=options(suggestions=['sum']), help="the method to collect losses")
        log_map = DictField(value={
            "loss": "loss"
            }, help="the output loss name")

    config = NestField(value=Config, converter=nest_converter)


@register("user_additional_loss_collect", "sum")
def loss_sum(losses: Dict[str, torch.Tensor], **args:Dict):
    """sum all losses
    Args:
        losses (List): list of loss
    Returns: 
        sum of losses
    """
    loss = sum([losses[key] for key in losses])
    return loss


@register("loss", "multi_loss")
class MultiLoss(nn.Module):
    """ This module is NotImplemented yet don't use it
    """
    def __init__(self, config: MultiLossConfig):
        super(MultiLoss, self).__init__()
        self.config = config.config
        self.loss_collect = register.get("user_additional_loss_collect", self.config.loss_collect)
        config_dict = config.to_dict()
        module_dict = {}
        for loss in config_dict:
            if loss in PROTECTED:
                continue
            module_dict[loss] = ConfigTool.get_leaf_module(register, config_register, "loss", config_dict[loss], init=True)
        self.losses = nn.ModuleDict(module_dict)

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

    def _calc(self, result, inputs, rt_config):
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
        for loss_module in self.losses:
            loss, log = self.losses[loss_module](result, inputs, rt_config)
            losses[loss_module] = loss
            log_loss.update(log)
        loss = self.loss_collect(losses=losses, rt_config=rt_config)
        if self.config.loss_collect != 'sum':
            log_loss.update({self.config.log_map.get("sum_loss", "sum_loss"): sum([losses[key] for key in losses])})
        log_loss.update({self.config.log_map['loss']: loss})
        return loss, log_loss

    def __call__(self, result, inputs, rt_config):
        """same as self.calc
        """
        return self._calc(result, inputs, rt_config)
