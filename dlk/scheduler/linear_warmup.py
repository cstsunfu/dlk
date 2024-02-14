# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch.optim as optim
from intc import (
    MISSING,
    AnyField,
    Base,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
)
from torch.optim.lr_scheduler import LambdaLR

from dlk.utils.register import register

from . import BaseScheduler, BaseSchedulerConfig


@cregister("scheduler", "linear_warmup")
class LinearWarmupScheduleConfig(BaseSchedulerConfig):
    """
    the linear warmup scheduler
    """


@register("scheduler", "linear_warmup")
class LinearWarmupSchedule(BaseScheduler):
    """linear warmup then linear decay"""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        config: LinearWarmupScheduleConfig,
        rt_config: Dict,
    ):
        super(LinearWarmupSchedule, self).__init__(optimizer, config, rt_config)
        self.config: LinearWarmupScheduleConfig = config

    def step_update(self, current_step: int):
        if current_step < self.config.num_warmup_steps:
            return float(current_step) / float(max(1, self.config.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - self.config.num_warmup_steps)),
        )

    def epoch_update(self, current_epoch: int):
        if current_epoch < self.config.num_warmup_steps:
            return float(current_epoch) / float(max(1, self.config.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_epochs - current_epoch)
            / float(max(1, self.num_training_epochs - self.config.num_warmup_steps)),
        )
