# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dlk.utils.register import register

from . import BaseScheduler, BaseSchedulerConfig

logger = logging.getLogger(__name__)


@cregister("scheduler", "cosine_restart")
class CosineRestartScheduleConfig(BaseSchedulerConfig):
    """
    the cosine restart scheduler
    """

    first_restart_step = IntField(
        value=-1,
        help="if the interval is `step`, the first restart step, if the interval is `epoch`, it means the first restart epoch",
    )
    mult_fact = IntField(
        value=1,
        minimum=0,
        help="the multiplier for the next restart interval, A factor increases after a restart. Default: 1.",
    )
    eta_min = FloatField(
        value=0.0, minimum=0.0, help="Minimum learning rate. Default: 0."
    )


@register("scheduler", "cosine_restart")
class CosineRestartSchedule(BaseScheduler):
    """CosineRestartSchedule"""

    def __init__(
        self, optimizer: optim.Optimizer, config: CosineRestartScheduleConfig, rt_config
    ):
        super(CosineRestartSchedule, self).__init__(optimizer, config, rt_config)
        self.config = config

    def get_scheduler(self) -> CosineAnnealingWarmRestarts:
        """return the initialized linear wramup then cos decay scheduler

        Returns:
            Schedule

        """
        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.first_restart_step,
            T_mult=self.config.mult_fact,
            eta_min=self.config.eta_min,
            last_epoch=-1,
        )
