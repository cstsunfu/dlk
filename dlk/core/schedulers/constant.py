from typing import Dict
import torch.nn as nn
from dlk.utils.config import BaseConfig
from . import scheduler_register, scheduler_config_register, BaseScheduler
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim


@scheduler_config_register("constant")
class ConstantScheduleConfig(BaseConfig):
    """
    {
        "config": {
            "last_epoch": -1
        },
        "_name": "constant",
    }
    """
    def __init__(self, config: Dict):
        super(ConstantScheduleConfig, self).__init__(config)
        config = config['config']
        self.last_epoch = config["last_epoch"]
        self.post_check(config, used=[
            "last_epoch",
        ])


@scheduler_register("constant")
class ConstantSchedule(BaseScheduler):
    def __init__(self, optimizer: optim.Optimizer, config: ConstantScheduleConfig):
        super(ConstantSchedule, self).__init__()
        self.config = config
        self.optimizer = optimizer

    def get_scheduler(self):
        """TODO: Docstring for get_scheduler.
        :returns: TODO
        """
        return LambdaLR(self.optimizer, lambda _: 1, last_epoch=self.config.last_epoch)
