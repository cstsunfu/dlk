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

"""schedulers"""
import importlib
import os
from typing import Dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules


@define
class BaseSchedulerConfig(BaseConfig):
    name = NameField(value="*@*", file=__file__, help="the base scheduler")
    @define
    class Config:
        num_training_epochs = IntField(value=0, checker=int_check(lower=0), help="the number of training epochs")
        epoch_training_steps = IntField(value=0, checker=int_check(lower=0), help="the number of training steps in one epoch")
        num_warmup_steps = NumberField(value=0.1, checker=number_check(lower=0), help="only for the scheduler with warmup, for the interval is `step`, the number of warmup steps, it can be float or int, if it's between 0.0-1.0, means the percentage of total steps, if it's larger than 1, means the number of steps. If the interval is `epoch`, it means the number of warmup epochs")
        num_training_steps = IntField(value=0, checker=int_check(lower=0), help="the number of training steps")
        last_epoch = IntField(value=-1, checker=int_check(lower=-1), help="the last epoch/step")
        interval = StrField(value="step", checker=str_check(["epoch", "step"]), help="the interval of scheduler")
    config = NestField(value=Config, converter=nest_converter)


class BaseScheduler(object):
    """interface for Schedule"""

    def __init__(self, optimizer: Optimizer, config: BaseSchedulerConfig, rt_config: Dict):
        """update dynamic parameters in self.config based the rt_config
        Args:
            rt_config: provide the current training status 
                >>> {
                >>>     "num_training_steps": self.num_training_steps,
                >>>     "epoch_training_steps": self.epoch_training_steps,
                >>>     "num_training_epochs": self.num_training_epochs
                >>> }
        Returns: None
        """
        super(BaseScheduler, self).__init__()
        self.config = config.config
        self.optimizer = optimizer
        self.config.num_training_steps = rt_config['num_training_steps']
        self.config.epoch_training_steps = rt_config['epoch_training_steps']
        self.config.num_training_epochs = rt_config['num_training_epochs']
        if self.config.num_warmup_steps < 1:
            if self.config.interval == "step":
                self.config.num_warmup_steps = int(self.config.num_warmup_steps * self.config.num_training_steps)
            else:
                assert self.config.interval == "epoch"
                self.config.num_warmup_steps = int(self.config.num_warmup_steps * self.config.num_training_epochs)

    def get_scheduler(self)->LambdaLR:
        """return the initialized scheduler

        Returns: 
            Schedule
        """
        if self.config.interval == "step":
            return LambdaLR(self.optimizer, self.step_update, last_epoch=self.config.last_epoch)
        else:
            assert self.config.interval == "epoch"
            return LambdaLR(self.optimizer, self.epoch_update, last_epoch=self.config.last_epoch)

    def step_update(self, current_step: int):
        raise NotImplementedError

    def epoch_update(self, current_epoch: int):
        raise NotImplementedError

    def __call__(self):
        """the same as self.get_scheduler()
        """
        return self.get_scheduler()


def import_schedulers(schedulers_dir, namespace):
    for file in os.listdir(schedulers_dir):
        path = os.path.join(schedulers_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            scheduler_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + scheduler_name)


# automatically import any Python files in the schedulers directory
schedulers_dir = os.path.dirname(__file__)
import_schedulers(schedulers_dir, "dlk.core.schedulers")
