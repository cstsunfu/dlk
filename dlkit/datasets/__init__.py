"""datasets"""

import importlib
import os
from typing import Callable, Dict, Type
from dlkit.utils.config import Config
from dlkit.utils.register import Register
import abc

dataset_config_register = Register("Dataset config register")
dataset_register = Register("Dataset register")


def import_datasets(datasets_dir, namespace):
    for file in os.listdir(datasets_dir):
        path = os.path.join(datasets_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            dataset_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + dataset_name)


# automatically import any Python files in the models directory
datasets_dir = os.path.dirname(__file__)
import_datasets(datasets_dir, "dlkit.datasets")
