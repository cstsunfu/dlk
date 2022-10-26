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
from . import scheduler_register, scheduler_config_register, BaseScheduler, BaseSchedulerConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim
from dlk.utils.logger import Logger
logger = Logger.get_logger()


@scheduler_config_register("cosine_restart")
class CosineRestartScheduleConfig(BaseSchedulerConfig):
    default_config = {
        "config": {
            "first_restart_step": -1, # you can just set first_restart_step or first_restart_epoch
            "first_restart_epoch": -1,
            "mult_fact": 1,
            },
        "_name": "cosine_restart",
    }
    """Config for CosineRestartSchedule

    Config Example: see default_config
    """
    def __init__(self, config: Dict):
        super(CosineRestartScheduleConfig, self).__init__(config)
        config = config['config']
        self.first_restart_epoch = config["first_restart_epoch"]
        self.first_restart_step = config["first_restart_step"]
        assert self.first_restart_epoch != -1 or self.first_restart_step != -1, "You must provide one of them"
        assert self.first_restart_epoch == -1 or self.first_restart_step == -1, "You must provide one of them"
        self.mult_fact = config['mult_fact']
        self.post_check(config, used=[
            "mult_fact",
            "first_restart_epoch",
            "first_restart_step",
        ])


@scheduler_register("cosine_restart")
class CosineRestartSchedule(BaseScheduler):
    """CosineRestartSchedule"""
    def __init__(self, optimizer: optim.Optimizer, config: CosineRestartScheduleConfig):
        super(CosineRestartSchedule, self).__init__()
        self.config = config
        self.optimizer = optimizer

    def get_scheduler(self)->CosineAnnealingWarmRestarts:
        """return the initialized linear wramup then cos decay scheduler

        Returns: 
            Schedule

        """
        epoch_training_steps = self.config.epoch_training_steps
        num_training_steps = self.config.num_training_steps
        if self.config.first_restart_step == -1:
            self.config.first_restart_step = self.config.first_restart_epoch * epoch_training_steps
        logger.warning(f"The calculated Total Traning Num is {num_training_steps}, the Restart Steps is {self.config.first_restart_step}. Please check it carefully.")

        return CosineAnnealingWarmRestarts(self.optimizer, self.config.first_restart_step, self.config.mult_fact, last_epoch=-1)
