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
from dlk.core.base_module import BaseModuleConfig
import torch.nn as nn


@loss_config_register("identity")
class IdentityLossConfig(BaseModuleConfig):
    """Config for IdentityLoss

    Config Example:
        >>> {
        >>>     config: {
        >>>         "schedule": [1],
        >>>         "scale": [1], # scale the loss for every schedule
        >>>         // "schedule": [0.3, 1.0], # can be a list or str
        >>>         // "scale": "[0.5, 1]",
        >>>         "loss": "loss", // the real loss from result['loss']
        >>>     },
        >>>     _name: "identity",
        >>> }
    """
    def __init__(self, config: Dict):
        super(IdentityLossConfig, self).__init__(config)
        config = config['config']

        self.scale = config['scale']
        self.schedule = config['schedule']
        self.loss = config['loss']

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
        self.post_check(config, used=[
            "loss",
            "schedule",
            "scale",
        ])

@loss_register("identity")
class IdentityLoss(object):
    """gather the loss and return when the loss is calc previor module like crf
    """
    def __init__(self, config: IdentityLossConfig):
        super(IdentityLoss, self).__init__()
        self.config = config

    def update_config(self, rt_config):
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
        loss = result[self.config.loss] * scale
        return loss

    def __call__(self, result, inputs, rt_config):
        """same as self.calc
        """
        return self.calc(result, inputs, rt_config)
