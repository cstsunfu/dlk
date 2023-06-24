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
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from . import BaseScheduler, BaseSchedulerConfig
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("scheduler", "constant")
@define
class ConstantScheduleConfig(BaseSchedulerConfig):
    name = NameField(value="constant", file=__file__, help="the constant scheduler")


@register("scheduler", "constant")
class ConstantSchedule(BaseScheduler):
    """no schedule
    """
    def __init__(self, optimizer: optim.Optimizer, config: ConstantScheduleConfig, rt_config):
        super(ConstantSchedule, self).__init__(optimizer, config, rt_config)

    def step_update(self, current_step: int):
        return 1

    def epoch_update(self, current_epoch: int):
        return 1
