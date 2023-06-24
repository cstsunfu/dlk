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
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim
from dlk.utils.logger import Logger
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules
from . import BaseScheduler, BaseSchedulerConfig
logger = Logger.get_logger()


@config_register("scheduler", "cosine_restart")
@define
class CosineRestartScheduleConfig(BaseSchedulerConfig):
    name = NameField(value="cosine_restart", file=__file__, help="the consine restart scheduler, it")
    @define
    class Config(BaseSchedulerConfig.Config):
        first_restart_step = IntField(value=-1, help="if the interval is `step`, the first restart step, if the interval is `epoch`, it means the first restart epoch")
        mult_fact = IntField(value=1, checker=int_check(lower=0), help="the multiplier for the next restart interval, A factor increases after a restart. Default: 1.")
        eta_min = FloatField(value=0.0, checker=float_check(lower=0.0), help="Minimum learning rate. Default: 0.")

    config = NestField(value=Config, converter=nest_converter)


@register("scheduler", "cosine_restart")
class CosineRestartSchedule(BaseScheduler):
    """CosineRestartSchedule"""
    def __init__(self, optimizer: optim.Optimizer, config: CosineRestartScheduleConfig, rt_config):
        super(CosineRestartSchedule, self).__init__(optimizer, config, rt_config)
        self.config: config.config

    def get_scheduler(self)->CosineAnnealingWarmRestarts:
        """return the initialized linear wramup then cos decay scheduler

        Returns: 
            Schedule

        """
        return CosineAnnealingWarmRestarts(self.optimizer, T_0=self.config.first_restart_step, T_mult=self.config.mult_fact, eta_min=self.config.eta_min, last_epoch=-1)
