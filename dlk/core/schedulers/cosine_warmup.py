from typing import Dict
import math
from dlk.utils.config import BaseConfig
from . import scheduler_register, scheduler_config_register, BaseScheduler
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim


@scheduler_config_register("cosine_warmup")
class CosineWarmupScheduleConfig(BaseConfig):
    """
    {
        "config": {
            "last_epoch": -1,
            "num_warmup_steps": 0,
            "num_training_steps": -1,
            "num_cycles": 0.5,
        },
        "_name": "cosine_warmup",
    }
    """
    def __init__(self, config: Dict):
        super(CosineWarmupScheduleConfig, self).__init__(config)
        config = config['config']
        self.last_epoch = config["last_epoch"]
        self.num_warmup_steps = config["num_warmup_steps"]
        self.num_training_steps = config["num_training_steps"]
        self.num_cycles = config['num_cycles']
        self.post_check(config, used=[
            "last_epoch",
            "num_warmup_steps",
            "num_training_steps",
            "num_cycles",
        ])


@scheduler_register("cosine_warmup")
class CosineWarmupSchedule(BaseScheduler):
    def __init__(self, optimizer: optim.Optimizer, config: CosineWarmupScheduleConfig):
        super(CosineWarmupSchedule, self).__init__()
        self.config = config
        self.optimizer = optimizer

    def get_scheduler(self):
        """TODO: Docstring for get_scheduler.
        :returns: TODO
        """
        num_training_steps = self.config.num_training_steps
        num_warmup_steps = self.config.num_warmup_steps
        if num_warmup_steps >0 and num_warmup_steps < 1:
            num_warmup_steps = int(num_warmup_steps * num_training_steps)
        last_epoch = self.config.last_epoch

        num_cycles = self.config.num_cycles


        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return LambdaLR(self.optimizer, lr_lambda, last_epoch)
