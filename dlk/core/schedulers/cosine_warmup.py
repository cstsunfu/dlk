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
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from . import BaseScheduler, BaseSchedulerConfig
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@config_register("scheduler", "cosine_warmup")
@define
class CosineWarmupScheduleConfig(BaseSchedulerConfig):
    name = NameField(value="cosine_warmup", file=__file__, help="the cosine_warmup scheduler")
    @define
    class Config(BaseSchedulerConfig.Config):
        num_cycles = FloatField(value=0.5, checker=float_check(0.0, suggestions=[0.5]), help="""0.5 is for cosine annealing, the num cycles of cosine warmup, you can use the code below to check the scheduler the
                                import math
                                num_cycles = 0.5
                                for progress in range(100):
                                    progress = progress / 100
                                    print(max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))
                                """)
    config = NestField(value=Config, converter=nest_converter)



@register("scheduler", "cosine_warmup")
class CosineWarmupSchedule(BaseScheduler):
    """CosineWarmupSchedule"""
    def __init__(self, optimizer: optim.Optimizer, config: CosineWarmupScheduleConfig, rt_config: Dict):
        super(CosineWarmupSchedule, self).__init__(optimizer, config, rt_config)
        self.config: CosineWarmupScheduleConfig.config

    def step_update(self, current_step: int):
        if current_step < self.config.num_warmup_steps:
            return float(current_step) / float(max(1, self.config.num_warmup_steps))
        progress = float(current_step - self.config.num_warmup_steps) / float(max(1, self.config.num_training_steps - self.config.num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.config.num_cycles) * 2.0 * progress)))

    def epoch_update(self, current_epoch: int):
        if current_epoch < self.config.num_warmup_steps:
            return float(current_epoch) / float(max(1, self.config.num_warmup_steps))
        progress = float(current_epoch - self.config.num_warmup_steps) / float(max(1, self.config.num_training_epochs - self.config.num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.config.num_cycles) * 2.0 * progress)))
