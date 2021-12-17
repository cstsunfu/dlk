# TODO: WIP Comming soon
# from typing import Dict
# import torch.nn as nn
# from . import scheduler_register, scheduler_config_register, BaseScheduler
# from torch.optim.lr_scheduler import LambdaLR
# import torch.optim as optim
# import copy

# # constant

# # cos warmup
# # def lr_lambda(current_step):
    # # if current_step < num_warmup_steps:
        # # return float(current_step) / float(max(1, num_warmup_steps))
    # # progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    # # return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


# # constant warmup
# # def lr_lambda(current_step: int):
    # # if current_step < num_warmup_steps:
        # # return float(current_step) / float(max(1.0, num_warmup_steps))
    # # return 1.0


# # linear warmup
# # def lr_lambda(current_step: int):
    # # if current_step < num_warmup_steps:
        # # return float(current_step) / float(max(1, num_warmup_steps))
    # # return max(
        # # 0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))


# @scheduler_config_register("constant")
# class ConstantScheduleConfig(object):
    # """
    # {
        # config: {
            # "last_epoch": -1
        # },
        # _name: "constant",
    # }
    # """
    # def __init__(self, config: Dict):
        # super(ConstantScheduleConfig, self).__init__()
        # config = config['config']
        # self.last_epoch = config["last_epoch"]


# @scheduler_register("constant")
# class ConstantSchedule(BaseScheduler):
    # def __init__(self, optimizer: optim.Optimizer, config: ConstantScheduleConfig):
        # super(ConstantSchedule, self).__init__()
        # self.config = config
        # self.optimizer = optimizer

    # def get_scheduler(self):
        # """TODO: Docstring for get_scheduler.
        # :returns: TODO
        # """
        # return LambdaLR(self.optimizer, lambda _: 1, last_epoch=self.config.last_epoch)
