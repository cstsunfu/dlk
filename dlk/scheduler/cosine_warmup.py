# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import math
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


@cregister("scheduler", "cosine_warmup")
class CosineWarmupScheduleConfig(BaseSchedulerConfig):
    """
    the cosine warmup scheduler
    """

    num_cycles = FloatField(
        value=0.5,
        minimum=0.0,
        suggestions=[0.5],
        help="""0.5 is for cosine annealing, the num cycles of cosine warmup, you can use the code below to check the scheduler the
            ```
                import math
                num_cycles = 0.5
                for progress in range(100):
                    progress = progress / 100
                    print(max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))
            ```
        """,
    )


@register("scheduler", "cosine_warmup")
class CosineWarmupSchedule(BaseScheduler):
    """CosineWarmupSchedule"""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        config: CosineWarmupScheduleConfig,
        rt_config: Dict,
    ):
        super(CosineWarmupSchedule, self).__init__(optimizer, config, rt_config)
        self.config: CosineWarmupScheduleConfig

    def step_update(self, current_step: int):
        if current_step < self.config.num_warmup_steps:
            return float(current_step) / float(max(1, self.config.num_warmup_steps))
        progress = float(current_step - self.config.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.config.num_warmup_steps)
        )
        return max(
            0.0,
            0.5
            * (
                1.0 + math.cos(math.pi * float(self.config.num_cycles) * 2.0 * progress)
            ),
        )

    def epoch_update(self, current_epoch: int):
        if current_epoch < self.config.num_warmup_steps:
            return float(current_epoch) / float(max(1, self.config.num_warmup_steps))
        progress = float(current_epoch - self.config.num_warmup_steps) / float(
            max(1, self.num_training_epochs - self.config.num_warmup_steps)
        )
        return max(
            0.0,
            0.5
            * (
                1.0 + math.cos(math.pi * float(self.config.num_cycles) * 2.0 * progress)
            ),
        )
