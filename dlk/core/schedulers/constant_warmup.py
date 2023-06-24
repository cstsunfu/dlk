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
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from . import BaseScheduler, BaseSchedulerConfig
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("scheduler", "constant_warmup")
@define
class ConstantWarmupScheduleConfig(BaseSchedulerConfig):
    name = NameField(value="constant_warmup", file=__file__, help="the constant_warmup scheduler")


@register("scheduler", "constant_warmup")
class ConstantWarmupSchedule(BaseScheduler):
    """ConstantWarmupSchedule"""
    def __init__(self, optimizer: optim.Optimizer, config: ConstantWarmupScheduleConfig, rt_config: Dict):
        super(ConstantWarmupSchedule, self).__init__(optimizer, config, rt_config)

    def step_update(self, current_step: int):
        if current_step < self.config.num_warmup_steps:
            return float(current_step) / float(max(1.0, self.config.num_warmup_steps))
        return 1.0

    def epoch_update(self, current_epoch: int):
        if current_epoch < self.config.num_warmup_steps:
            return float(current_epoch) / float(max(1.0, self.config.num_warmup_steps))
        return 1.0
