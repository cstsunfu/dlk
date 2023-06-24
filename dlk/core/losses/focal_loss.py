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

from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from dlk.utils.logger import Logger
import torch
from . import BaseLoss, BaseLossConfig
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules

logger = Logger.get_logger()


@config_register("loss", "focal_loss")
@define
class FocalLossConfig(BaseLossConfig):
    name = NameField(value="focal_loss", file=__file__, help="the focal_loss loss")
    @define
    class Config(BaseLossConfig.Config):
        weight = ListField(value=None, checker=suggestions([None]), help="the list of weights of every class")
        ignore_index = IntField(value=-100, checker=int_check(), help="the ignore index")
        gamma = FloatField(value=1.0, checker=float_check(), help="the gamma of focal loss")
    config = NestField(value=Config, converter=nest_converter)


@register("loss", "focal_loss")
class FocalLoss(BaseLoss):
    """for multi class classification
    """
    def __init__(self, config: FocalLossConfig):
        super(FocalLoss, self).__init__(config)
        self.config: FocalLossConfig.Config
        self.weight: torch.Tensor = None
        if self.config.weight:
            self.weight = torch.tensor(self.config.weight, dtype=torch.float)


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
        return loss, {self.config.log_map['loss']: loss}
