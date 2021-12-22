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
from dlk.utils.config import BaseConfig
from . import scheduler_register, scheduler_config_register, BaseScheduler
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim


@scheduler_config_register("constant_warmup")
class ConstantWarmupScheduleConfig(BaseConfig):
    """Config for ConstantWarmupSchedule

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "last_epoch": -1,
        >>>         "num_warmup_steps": 0,
        >>>     },
        >>>     "_name": "constant_warmup",
        >>> }
    """
    def __init__(self, config: Dict):
        super(ConstantWarmupScheduleConfig, self).__init__(config)
        config = config['config']
        self.last_epoch = config["last_epoch"]
        self.num_warmup_steps = config["num_warmup_steps"]
        self.post_check(config, used=[
            "last_epoch",
            "num_warmup_steps",
        ])


@scheduler_register("constant_warmup")
class ConstantWarmupSchedule(BaseScheduler):
    """ConstantWarmupSchedule"""
    def __init__(self, optimizer: optim.Optimizer, config: ConstantWarmupScheduleConfig):
        super(ConstantWarmupSchedule, self).__init__()
        self.config = config
        self.optimizer = optimizer

    def get_scheduler(self)->LambdaLR:
        """return the initialized linear wramup then constant scheduler

        Returns: 
            Schedule

        """
        num_warmup_steps = self.config.num_warmup_steps
        last_epoch = self.config.last_epoch

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0

        return LambdaLR(self.optimizer, lr_lambda, last_epoch=last_epoch)
