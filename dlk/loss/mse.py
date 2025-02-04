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


@cregister("loss", "mse")
class MSELossConfig(BaseLossConfig):
    """the MEAN SQUARE ERROR loss(MSE AKA L2 loss)"""

    pass


@register("loss", "mse")
class MSELoss(BaseLoss):
    """mse loss for regression, distill, etc."""

    def __init__(self, config: MSELossConfig):
        super(MSELoss, self).__init__(config)
        self.config: MSELossConfig
        self.mse = nn.MSELoss(
            reduction=self.config.reduction
        )  # we will use masked_select, and apply reduction=batchmean (sum/batch_size)

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
        pred = result[self.pred_name]
        target = inputs[self.truth_name]
        batch_size = target.shape[0]
        loss = self.mse(pred, target) * scale / batch_size  # batch mean
        return loss, {self.config.log_map.loss: loss}
