"""losses"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlk.utils.register import Register

loss_config_register = Register("Model config register.")
loss_register = Register("Model register.")

def import_losses(losses_dir, namespace):
    for file in os.listdir(losses_dir):
        path = os.path.join(losses_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            loss_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + loss_name)


# automatically import any Python files in the losses directory
losses_dir = os.path.dirname(__file__)
import_losses(losses_dir, "dlk.core.losses")
