# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict

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
    dataclass,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from dlk.utils.import_module import import_module_dir


@dataclass
class BaseSchedulerConfig(Base):
    """the base scheduler"""

    num_warmup_steps = FloatField(
        value=0.1,
        minimum=0.0,
        help="""
            only for the scheduler with warmup.
            for the interval is `step`,
            the number of warmup steps,
            it can be float or int, 
            if it's between 0.0-1.0, means the percentage of total steps,
            if it's larger than 1, means the number of steps.
            If the interval is `epoch`,
            it means the number of warmup epochs
        """,
    )
    last_epoch = IntField(value=-1, minimum=-1, help="the last epoch/step")
    interval = StrField(
        value="step", options=["epoch", "step"], help="the interval of scheduler"
    )


class BaseScheduler(object):
    """interface for Schedule"""

    def __init__(
        self, optimizer: Optimizer, config: BaseSchedulerConfig, rt_config: Dict
    ):
        """update dynamic parameters in self.config based the rt_config
        Args:
            rt_config: provide the current training status
                >>> {
                >>>     "num_training_steps": self.num_training_steps,
                >>>     "num_training_epochs": self.num_training_epochs
                >>> }
        Returns: None
        """
        super(BaseScheduler, self).__init__()
        self.config = config
        self.optimizer = optimizer
        self.num_training_steps = rt_config["num_training_steps"]
        self.num_training_epochs = rt_config["num_training_epochs"]
        if self.config.num_warmup_steps < 1:
            if self.config.interval == "step":
                self.config.num_warmup_steps = int(
                    self.config.num_warmup_steps * self.num_training_steps
                )
            else:
                assert self.config.interval == "epoch"
                self.config.num_warmup_steps = int(
                    self.config.num_warmup_steps * self.num_training_epochs
                )

    def get_scheduler(self) -> LambdaLR:
        """return the initialized scheduler

        Returns:
            Schedule
        """
        if self.config.interval == "step":
            return LambdaLR(
                self.optimizer, self.step_update, last_epoch=self.config.last_epoch
            )
        else:
            assert self.config.interval == "epoch"
            return LambdaLR(
                self.optimizer, self.epoch_update, last_epoch=self.config.last_epoch
            )

    def step_update(self, current_step: int):
        raise NotImplementedError

    def epoch_update(self, current_epoch: int):
        raise NotImplementedError

    def __call__(self):
        """the same as self.get_scheduler()"""
        return self.get_scheduler()


scheduler_dir = os.path.dirname(__file__)
import_module_dir(scheduler_dir, "dlk.scheduler")
