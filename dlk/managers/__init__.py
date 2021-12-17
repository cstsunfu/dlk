"""managers"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlk.utils.register import Register

manager_config_register = Register("Manager config register.")
manager_register = Register("Manager register.")

def import_managers(managers_dir, namespace):
    for file in os.listdir(managers_dir):
        path = os.path.join(managers_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            manager_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + manager_name)


# automatically import any Python files in the managers directory
managers_dir = os.path.dirname(__file__)
import_managers(managers_dir, "dlk.managers")
