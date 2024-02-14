# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

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

from dlk.utils.register import register

from . import BaseLoss, BaseLossConfig


@cregister("loss", "identity")
class IdentityLossConfig(BaseLossConfig):
    """do not calc loss in this step, but calculated in previous"""

    loss = StrField(value="loss", help="the pre calculated loss result")


@register("loss", "identity")
class IdentityLoss(BaseLoss):
    """do not calc loss in this step, but calculated in previous"""

    def __init__(self, config: IdentityLossConfig):
        super(IdentityLoss, self).__init__(config)
        self.config: IdentityLossConfig

    def _calc(self, result, inputs, rt_config, scale):
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
            scale: the scale rate for current stage
        Returns:
            loss

        """

        if rt_config["current_step"] > self.config.schedule[self.current_stage]:
            self.current_stage += 1
        scale = self.config.scale[self.current_stage]
        loss = result[self.config.loss] * scale
        return loss, {self.config.log_map.loss: loss}
