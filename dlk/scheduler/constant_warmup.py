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


@cregister("scheduler", "constant_warmup")
class ConstantWarmupScheduleConfig(BaseSchedulerConfig):
    """the constant scheduler begin with a warmup"""


@register("scheduler", "constant_warmup")
class ConstantWarmupSchedule(BaseScheduler):
    """ConstantWarmupSchedule"""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        config: ConstantWarmupScheduleConfig,
        rt_config: Dict,
    ):
        super(ConstantWarmupSchedule, self).__init__(optimizer, config, rt_config)

    def step_update(self, current_step: int):
        if current_step < self.config.num_warmup_steps:
            return float(current_step) / float(max(1.0, self.config.num_warmup_steps))
        return 1.0

    def epoch_update(self, current_epoch: int):
        if current_epoch < self.config.num_warmup_steps:
            return float(current_epoch) / float(max(1.0, self.config.num_warmup_steps))
        return 1.0
