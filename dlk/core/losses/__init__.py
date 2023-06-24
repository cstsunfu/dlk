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

"""losses"""
import importlib
import os
from typing import Dict
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules
import torch.nn as nn


@define
class BaseLossConfig(BaseConfig):
    name = NameField(value="*@*", file=__file__, help="the loss name")
    @define
    class Config:
        schedule = ListField(value=[1], help="the schedule of the loss, works with scale")
        scale = ListField(value=[1], help="the scale of the loss for every schedule stage")
        pred_truth_pair = ListField(value=[], checker=suggestions([["logits", "label_ids"]]), help="it's a tuple include two values, [predit_logits, truth_target] pair. If you have more than one pair, you should use the `multi_loss` module and provide this module as the submodule.")
        reduction = StrField(value="mean", checker=options(['mean', 'sum', 'none']), help="the reduction method, support 'mean', 'sum', 'none'")
        log_map = DictField(value={
            "loss": "loss"
            }, help="the output loss name")

    config = NestField(value=Config, converter=nest_converter)


class BaseLoss(object):
    def __init__(self, config: BaseLossConfig):
        super(BaseLoss, self).__init__()
        self.config = config.config
        assert len(self.config.pred_truth_pair) == 2
        self.pred_name, self.truth_name = self.config.pred_truth_pair

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
        raise NotImplementedError

    def __call__(self, result, inputs, rt_config):
        """same as self.calc
        """
        if rt_config['current_step']>self.config.schedule[self.current_stage]:
            self.current_stage += 1
        scale = self.config.scale[self.current_stage]
        return self._calc(result, inputs, rt_config, scale)


def import_losses(losses_dir, namespace):
    for file in os.listdir(losses_dir):
        path = os.path.join(losses_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            loss_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + loss_name)


# automatically import any Python files in the losses directory
losses_dir = os.path.dirname(__file__)
import_losses(losses_dir, "dlk.core.losses")
