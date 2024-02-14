# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch.nn as nn
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


@cregister("scheduler", "constant")
class ConstantScheduleConfig(BaseSchedulerConfig):
    """the constant scheduler"""

    pass


@register("scheduler", "constant")
class ConstantSchedule(BaseScheduler):
    """no schedule"""

    def __init__(
        self, optimizer: optim.Optimizer, config: ConstantScheduleConfig, rt_config
    ):
        super(ConstantSchedule, self).__init__(optimizer, config, rt_config)

    def step_update(self, current_step: int):
        return 1

    def epoch_update(self, current_epoch: int):
        return 1
