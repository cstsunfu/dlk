# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
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

logger = logging.getLogger(__name__)


@cregister("loss", "focal_loss")
class FocalLossConfig(BaseLossConfig):
    """the focal_loss loss config"""

    weight = ListField(
        value=None,
        additions=[None],
        help="the list of weights of every class",
    )
    ignore_index = IntField(value=-100, help="the ignore index")
    gamma = FloatField(value=1.0, help="the gamma of focal loss")


@register("loss", "focal_loss")
class FocalLoss(BaseLoss):
    """for multi class classification"""

    def __init__(self, config: FocalLossConfig):
        super(FocalLoss, self).__init__(config)
        self.config: FocalLossConfig
        if self.config.weight:
            self.weight = torch.tensor(self.config.weight, dtype=torch.float)
        else:
            self.weight = None

    def focal_loss(self, pred, target):
        """calc focal loss

        Args:
            pred: NxC for logits
            target: N for target

        Returns:
            loss

        """
        # drop the ignore target
        mask = target != self.config.ignore_index
        target = target[mask]
        pred = pred[mask]
        # [N, 1]
        target = target.unsqueeze(-1)
        # [N, C]
        pt = F.softmax(pred, dim=-1)
        logpt = F.log_softmax(pred, dim=-1)
        # [N]
        pt = pt.gather(1, target).squeeze(-1)
        logpt = logpt.gather(1, target).squeeze(-1)

        if self.weight is not None:
            # [N] at[i] = weight[target[i]]
            at = self.weight.gather(0, target.squeeze(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.config.gamma * logpt
        if self.config.reduction == "none":
            return loss
        if self.config.reduction == "mean":
            return loss.mean()
        return loss.sum()

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
            scale: the scale rate for the loss

        Returns:
            loss

        """
        pred = result[self.pred_name]
        target = inputs[self.truth_name]
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1)

        loss = self.focal_loss(pred, target) * scale
        return loss, {self.config.log_map.loss: loss}
