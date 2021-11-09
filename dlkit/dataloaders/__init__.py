"""dataloaders"""

import importlib
import os
from typing import Callable, Dict, Type
from dlkit.utils.config import Config
from dlkit.utils.register import Register
import abc

dataloader_config_register = Register("Dataloader config register")
dataloader_register = Register("Dataloader register")


def import_dataloaders(dataloaders_dir, namespace):
    for file in os.listdir(dataloaders_dir):
        path = os.path.join(dataloaders_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            dataloader_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + dataloader_name)


# automatically import any Python files in the models directory
dataloaders_dir = os.path.dirname(__file__)
import_dataloaders(dataloaders_dir, "dlkit.dataloaders")
