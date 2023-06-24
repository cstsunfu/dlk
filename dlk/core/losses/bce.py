# Copyright 2021 cstsunfu. All rights reserved.
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
from packaging import version
import torch.nn as nn
import torch
from . import BaseLoss, BaseLossConfig
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("loss", "bce")
@define
class BCEWithLogitsLossConfig(BaseLossConfig):
    name = NameField(value="bce", file=__file__, help="the bce loss")

@register("loss", "bce")
class BCEWithLogitsLoss(BaseLoss):
    """binary crossentropy for bi-class classification
    """
    def __init__(self, config: BCEWithLogitsLossConfig):
        super(BCEWithLogitsLoss, self).__init__(config)
        self.bce = nn.BCEWithLogitsLoss(reduction=self.config.reduction)

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
        loss = self.bce(torch.sigmoid(pred), target) * scale
        return loss, {self.config.log_map['loss']: loss}
