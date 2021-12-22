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


@scheduler_config_register("rec_decay")
class RecDecayScheduleConfig(BaseConfig):
    """Config for RecDecaySchedule

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "last_epoch": -1,
        >>>         "num_training_steps": -1,
        >>>         "decay": 0.05,
        >>>         "epoch_training_steps": -1,
        >>>     },
        >>>     "_name": "rec_decay",
        >>> }

    the lr=lr*1/(1+decay)
    """
    def __init__(self, config: Dict):
        super(RecDecayScheduleConfig, self).__init__(config)
        config = config['config']
        self.last_epoch = config["last_epoch"]
        self.epoch_training_steps = config["epoch_training_steps"]
        self.decay = config["decay"]
        self.num_training_steps = config["num_training_steps"]
        self.post_check(config, used=[
            "last_epoch",
            "num_training_steps",
            "decay",
            "epoch_training_steps",
        ])


@scheduler_register("rec_decay")
class RecDecaySchedule(BaseScheduler):
    """lr=lr*1/(1+decay)
    """
    def __init__(self, optimizer: optim.Optimizer, config: RecDecayScheduleConfig):
        super(RecDecaySchedule, self).__init__()
        self.config = config
        self.optimizer = optimizer

    def get_scheduler(self):
        """return the initialized rec_decay scheduler

        lr=lr*1/(1+decay)

        Returns: 
            Schedule

        """
        num_training_steps = self.config.num_training_steps
        epoch_training_steps = self.config.epoch_training_steps
        decay = self.config.decay
        last_epoch = self.config.last_epoch
        logger.warning(f"The calculated Total Traning Num is {num_training_steps}, the Epoch training Steps is {epoch_training_steps}. Please check it carefully.")

        def lr_lambda(current_step: int):
            cur_epoch = (current_step+1)//epoch_training_steps if epoch_training_steps!=0 else 0
            # return 1/(1+decay*cur_epoch)
            return 1/((1+decay)**cur_epoch)
        return LambdaLR(self.optimizer, lr_lambda, last_epoch)
