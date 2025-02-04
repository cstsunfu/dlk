# Copyright cstsunfu.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict

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
from packaging import version

from dlk.utils.register import register

from . import BaseLoss, BaseLossConfig

logger = logging.getLogger(__name__)


@cregister("loss", "cross_entropy")
class CrossEntropyLossConfig(BaseLossConfig):
    """the cross_entropy loss"""

    weight = ListField(
        value=None,
        additions=[None],
        help="the list of weights of every class",
    )
    label_smoothing = FloatField(
        value=0.0, minimum=0.0, maximum=1.0, help="the label smoothing"
    )
    ignore_index = IntField(value=-100, help="the ignore index")


@register("loss", "cross_entropy")
class CrossEntropyLoss(BaseLoss):
    """for multi class classification"""

    def __init__(self, config: CrossEntropyLossConfig):
        super(CrossEntropyLoss, self).__init__(config)
        self.config: CrossEntropyLossConfig
        if self.config.weight:
            weight = torch.tensor(self.config.weight, dtype=torch.float)
        else:
            weight = None
        if version.parse(torch.__version__) >= version.parse("1.10"):
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=self.config.ignore_index,
                reduction=self.config.reduction,
                label_smoothing=self.config.label_smoothing,
            )
        else:
            if self.config.label_smoothing:
                logger.info("Torch version is <1.10, so ignore label_smoothing")
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=self.config.ignore_index,
                reduction=self.config.reduction,
            )

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
        loss = self.cross_entropy(pred, target) * scale
        return loss, {self.config.log_map.loss: loss}
