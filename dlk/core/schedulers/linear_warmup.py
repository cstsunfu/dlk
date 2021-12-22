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
from dlk.utils.logger import Logger
import torch.optim as optim
logger = Logger.get_logger()


@scheduler_config_register("linear_warmup")
class LinearWarmupScheduleConfig(BaseConfig):
    """
    Config Example:
        >>> {
        >>>     "config": {
        >>>         "last_epoch": -1,
        >>>         "num_warmup_steps": 0,
        >>>         "num_training_steps": -1,
        >>>     },
        >>>     "_name": "linear_warmup",
        >>> }
    """
    def __init__(self, config: Dict):
        super(LinearWarmupScheduleConfig, self).__init__(config)
        config = config['config']
        self.last_epoch = config["last_epoch"]
        self.num_warmup_steps = config["num_warmup_steps"]
        self.num_training_steps = config["num_training_steps"]
        self.post_check(config, used=[
            "last_epoch",
            "num_warmup_steps",
            "num_training_steps",
        ])


@scheduler_register("linear_warmup")
class LinearWarmupSchedule(BaseScheduler):
    """linear warmup then linear decay"""
    def __init__(self, optimizer: optim.Optimizer, config: LinearWarmupScheduleConfig):
        super(LinearWarmupSchedule, self).__init__()
        self.config = config
        self.optimizer = optimizer

    def get_scheduler(self)->LambdaLR:
        """return the initialized linear wramup then linear decay scheduler

        Returns: 
            Schedule

        """
        num_training_steps = self.config.num_training_steps
        num_warmup_steps = self.config.num_warmup_steps
        if num_warmup_steps >0 and num_warmup_steps < 1:
            num_warmup_steps = int(num_warmup_steps * num_training_steps)
        last_epoch = self.config.last_epoch
        logger.warning(f"The calculated Total Traning Num is {num_training_steps}, the Num Warmup Steps is {num_warmup_steps}. Please check it carefully.")

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(self.optimizer, lr_lambda, last_epoch)
