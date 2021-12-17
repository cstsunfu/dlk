"""schedulers"""
import importlib
import os
from dlk.utils.register import Register
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


scheduler_config_register = Register("Schedule config register.")
scheduler_register = Register("Schedule register.")


class BaseScheduler(object):
    """docstring for BaseSchedule"""

    def get_scheduler(self):
        """TODO: Docstring for get_scheduler.
        :returns: TODO

        """
        raise NotImplementedError

    def __call__(self):
        """TODO: Docstring for __call__.

        :arg1: TODO
        :returns: TODO

        """
        return self.get_scheduler()


def import_schedulers(schedulers_dir, namespace):
    for file in os.listdir(schedulers_dir):
        path = os.path.join(schedulers_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            scheduler_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + scheduler_name)


# automatically import any Python files in the schedulers directory
schedulers_dir = os.path.dirname(__file__)
import_schedulers(schedulers_dir, "dlk.core.schedulers")
