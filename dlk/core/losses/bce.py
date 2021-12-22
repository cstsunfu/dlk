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
from . import loss_register, loss_config_register
from packaging import version
import torch.nn as nn
from dlk.utils.logger import Logger
from dlk.core.base_module import BaseModuleConfig
import torch

logger = Logger.get_logger()

@loss_config_register("bce")
class BCEWithLogitsLossConfig(BaseModuleConfig):
    """Config for BCEWithLogitsLoss

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "pred_truth_pair": [], # len(.) == 2, the 1st is the pred_name, 2nd is truth_name in __call__ inputs
        >>>         "schedule": [1],
        >>>         "masked_select": null, // if provide, only select the masked(=1) data
        >>>         "scale": [1], # scale the loss for every schedule stage
        >>>         // "schdeule": [0.3, 1.0], # can be a list or str
        >>>         // "scale": "[0.5, 1]",
        >>>     },
        >>>     "_name": "bce",
        >>> }
    """
    def __init__(self, config: Dict):
        super(BCEWithLogitsLossConfig, self).__init__(config)
        config = config['config']

        self.scale = config['scale']
        self.schedule = config['schedule']

        if isinstance(self.scale, str):
            self.scale = eval(self.scale)
        if isinstance(self.schedule, str):
            self.schedule = eval(self.schedule)

        if not isinstance(self.scale, list):
            assert isinstance(float(self.scale), float)
            self.scale = [self.scale]
        if not isinstance(self.schedule, list):
            assert isinstance(float(self.schedule), float)
            self.schedule = [self.schedule]
        assert len(self.schedule) == len(self.scale)
        assert self.schedule[-1] - 1 < 0.00001

        self.pred_truth_pair = config['pred_truth_pair']
        if not self.pred_truth_pair:
            raise PermissionError(f"You must provide the pred_truth_pair for loss.")
        self.masked_select = config['masked_select']
        self.post_check(config, used=[
            "pred_truth_pair",
            "masked_select",
            "schedule",
            "scale",
        ])


@loss_register("bce")
class BCEWithLogitsLoss(object):
    """binary crossentropy for bi-class classification
    """
    def __init__(self, config: BCEWithLogitsLossConfig):
        super(BCEWithLogitsLoss, self).__init__()
        self.config = config
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def update_config(self, rt_config: Dict):
        """callback for imodel to update the total steps and epochs

        when init the loss module, the total step and epoch is not known, when all data ready, the imodel update the value for loss module

        Args:
            rt_config: { "total_steps": self.num_training_steps, "total_epochs": self.num_training_epochs}

        Returns: 
            None

        """
        self.current_stage = 0
        self.config.schedule = [rt_config['total_steps']*i for i in self.config.schedule]

    def calc(self, result, inputs, rt_config):
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
        if rt_config['current_step']>self.config.schedule[self.current_stage]:
            self.current_stage += 1
        scale = self.config.scale[self.current_stage]
        pred_name, truth_name = self.config.pred_truth_pair
        pred = result[pred_name]
        target = inputs[truth_name]
        batch_size = target.shape[0]
        if self.config.masked_select is not None:
            pred = torch.masked_select(pred, inputs[self.config.masked_select])
            target = torch.masked_select(target, inputs[self.config.masked_select])
        loss = self.bce(torch.sigmoid(pred), target) * scale
        return loss

    def __call__(self, result, inputs, rt_config):
        """same as self.calc
        """
        return self.calc(result, inputs, rt_config)
