"""optimizers"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlkit.utils.register import Register
import torch.optim as optim


optimizer_config_register = Register("Optimizer config register.")
optimizer_register = Register("Optimizer register.")

optimizer_map = Register("Optimizer Map")

optimizer_map.register('adamw')(optim.AdamW)
optimizer_map.register('sgd')(optim.SGD)


def import_optimizers(optimizers_dir, namespace):
    for file in os.listdir(optimizers_dir):
        path = os.path.join(optimizers_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            optimizer_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + optimizer_name)


# automatically import any Python files in the optimizers directory
optimizers_dir = os.path.dirname(__file__)
import_optimizers(optimizers_dir, "dlkit.core.optimizers")
