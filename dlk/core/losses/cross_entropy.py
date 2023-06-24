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
from packaging import version
from dlk.utils.logger import Logger
import torch
from . import BaseLoss, BaseLossConfig
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules

logger = Logger.get_logger()


@config_register("loss", "cross_entropy")
@define
class CrossEntropyLossConfig(BaseLossConfig):
    name = NameField(value="cross_entropy", file=__file__, help="the cross_entropy loss")
    @define
    class Config(BaseLossConfig.Config):
        weight = ListField(value=None, checker=suggestions([None]), help="the list of weights of every class")
        label_smoothing = FloatField(value=0.0, checker=float_check(0.0, 1.0), help="the label smoothing")
        ignore_index = IntField(value=-100, checker=int_check(), help="the ignore index")
    config = NestField(value=Config, converter=nest_converter)

@register("loss", "cross_entropy")
class CrossEntropyLoss(BaseLoss):
    """for multi class classification
    """
    def __init__(self, config: CrossEntropyLossConfig):
        super(CrossEntropyLoss, self).__init__(config)
        self.config: CrossEntropyLossConfig.Config
        weight: torch.Tensor = self.config.weight
        if self.config.weight:
            weight = torch.tensor(self.config.weight, dtype=torch.float)
        if (version.parse(torch.__version__)>=version.parse("1.10")):
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=self.config.ignore_index,
                reduction=self.config.reduction,
                label_smoothing=self.config.label_smoothing
            )
        else:
            if self.config.label_smoothing:
                logger.info("Torch version is <1.10, so ignore label_smoothing")
            self.cross_entropy = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=self.config.ignore_index,
                reduction=self.config.reduction
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
        return loss, {self.config.log_map['loss']: loss}
