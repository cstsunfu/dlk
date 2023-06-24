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

import torch.nn as nn
from . import BaseLoss, BaseLossConfig
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("loss", "mse")
@define
class MSELossConfig(BaseLossConfig):
    name = NameField(value="mse", file=__file__, help="the mse loss")


@register("loss", "mse")
class MSELoss(BaseLoss):
    """ mse loss for regression, distill, etc.
    """
    def __init__(self, config: MSELossConfig):
        super(MSELoss, self).__init__(config)
        self.config: MSELossConfig.Config
        self.mse = nn.MSELoss(reduction=self.config.reduction) # we will use masked_select, and apply reduction=batchmean (sum/batch_size)

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
        loss = self.mse(pred, target) * scale / batch_size # batch mean
        return loss, {self.config.log_map['loss']: loss}
