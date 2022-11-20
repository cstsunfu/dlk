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
from . import scheduler_register, scheduler_config_register, BaseScheduler, BaseSchedulerConfig
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim


@scheduler_config_register("constant")
class ConstantScheduleConfig(BaseSchedulerConfig):
    default_config = {
            "config": {
                },
            "_name": "constant",
            }
    """Config for ConstantSchedule

    Config Example:
        default_config
    """
    def __init__(self, config: Dict):
        super(ConstantScheduleConfig, self).__init__(config)
        config = config['config']
        self.post_check(config, used=[
        ])


@scheduler_register("constant")
class ConstantSchedule(BaseScheduler):
    """no schedule
    """
    def __init__(self, optimizer: optim.Optimizer, config: ConstantScheduleConfig):
        super(ConstantSchedule, self).__init__()
        self.config = config
        self.optimizer = optimizer

    def get_scheduler(self):
        """return the initialized constant scheduler

        Returns: 
            Schedule

        """
        return LambdaLR(self.optimizer, lambda _: 1, last_epoch=-1)
