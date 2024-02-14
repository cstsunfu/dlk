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


@cregister("scheduler", "warmup_rec_decay")
class WarmupRecDecayScheduleConfig(BaseSchedulerConfig):
    """
    the warmup_rec_decay scheduler, lr=lr*1/(1+decay)
    """

    decay = FloatField(value=0.05, help="the decay rate, lr=lr*1/(1+decay)")
    interval = StrField(
        value="epoch", options=["epoch"], help="the interval of scheduler"
    )


@register("scheduler", "rec_decay")
class WarmupRecDecaySchedule(BaseScheduler):
    """lr=lr*1/(1+decay)"""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        config: WarmupRecDecayScheduleConfig,
        rt_config: Dict,
    ):
        super(WarmupRecDecaySchedule, self).__init__(optimizer, config, rt_config)
        self.config: WarmupRecDecayScheduleConfig = config

    def step_update(self, current_step: int):
        raise PermissionError("step_update is not supported in WarmupRecDecaySchedule")

    def epoch_update(self, current_epoch: int):
        if current_epoch < self.config.num_warmup_steps:
            return float(current_epoch) / float(max(1, self.config.num_warmup_steps))
        return 1 / ((1 + self.config.decay) ** current_epoch)
