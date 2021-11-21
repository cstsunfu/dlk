"""schedules"""
import importlib
import os
from dlkit.utils.register import Register
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


schedule_config_register = Register("Schedule config register.")
schedule_register = Register("Schedule register.")

schedule_map = Register("Schedule Map")


@schedule_map("constant")
def get_constant_schedule(optimizer: Optimizer, **config):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        **config include:
            last_epoch (:obj:`int`, `optional`, defaults to -1):
                The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    num_warmup_steps = config.get("num_warmup_steps")
    last_epoch = config.get("last_epoch", -1)
    return LambdaLR(optimizer, lambda _: 1, last_epoch=config.get("last_epoch", -1))


@schedule_map("constant_warmup")
def get_constant_schedule_with_warmup(optimizer: Optimizer, **config):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        **config include:
            last_epoch (:obj:`int`, `optional`, defaults to -1):
            num_warmup_steps (:obj:`int`):
                The number of steps for the warmup phase.
            last_epoch (:obj:`int`, `optional`, defaults to -1):
                The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    num_warmup_steps = config.get("num_warmup_steps")
    last_epoch = config.get("last_epoch", -1)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


@schedule_map("linear_warmup")
def get_linear_schedule_with_warmup(optimizer, **config):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        **config include:
            num_warmup_steps (:obj:`int`):
                The number of steps for the warmup phase.
            num_training_steps (:obj:`int`):
                The total number of training steps.
            last_epoch (:obj:`int`, `optional`, defaults to -1):
                The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    num_warmup_steps = config.get("num_warmup_steps")
    num_training_steps = config.get("num_training_steps")
    last_epoch = config.get("last_epoch", -1)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@schedule_map("cosine_warmup")
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, **config
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        **config include:
            num_warmup_steps (:obj:`int`):
                The number of steps for the warmup phase.
            num_training_steps (:obj:`int`):
                The total number of training steps.
            num_cycles (:obj:`float`, `optional`, defaults to 0.5):
                The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
                following a half-cosine).
            last_epoch (:obj:`int`, `optional`, defaults to -1):
                The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    num_warmup_steps = config.get("num_warmup_steps")
    num_training_steps = config.get("num_training_steps")
    num_cycles = config.get("num_cycles", 0.5)
    last_epoch = config.get("last_epoch", -1)


    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def import_schedules(schedules_dir, namespace):
    for file in os.listdir(schedules_dir):
        path = os.path.join(schedules_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            schedule_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + schedule_name)


# automatically import any Python files in the schedules directory
schedules_dir = os.path.dirname(__file__)
import_schedules(schedules_dir, "dlkit.schedules")
