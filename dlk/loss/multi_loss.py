# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
import torch.nn as nn
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
)

from dlk.utils.register import register, register_module_name


@cregister("loss", "multi_loss")
class MultiLossConfig(Base):
    """multiple loss"""

    loss_collect = StrField(
        value="sum",
        help="the method to collect losses",
    )

    class LogMap:
        loss = StrField(value="loss", help="the output loss name")

    log_map = NestField(value=LogMap, help="the output loss name")
    submodule = SubModule(value={}, help="the multible loss modules")


@register("additional_loss_collect", "sum")
def loss_sum(losses: Dict[str, torch.Tensor], **args: Dict):
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
    """This module is NotImplemented yet don't use it"""

    def __init__(self, config: MultiLossConfig):
        super(MultiLoss, self).__init__()
        self.config = config
        self.loss_collect = register.get(
            "additional_loss_collect", self.config.loss_collect
        )
        loss_configs = config._get_named_modules("loss")

        self.losses = nn.ModuleDict(
            {
                loss_name: register.get(
                    "loss", register_module_name(loss_config._module_name)
                )(loss_config)
                for loss_name, loss_config in loss_configs.items()
            }
        )

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
        log_loss.update({self.config.log_map.loss: loss})
        return loss, log_loss

    def __call__(self, result, inputs, rt_config):
        """same as self.calc"""
        return self._calc(result, inputs, rt_config)
