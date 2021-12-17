"""initmethods"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlk.utils.register import Register

initmethod_config_register = Register("InitMethod config register.")
initmethod_register = Register("InitMethod register.")


def import_initmethods(initmethods_dir, namespace):
    for file in os.listdir(initmethods_dir):
        path = os.path.join(initmethods_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            initmethod_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + initmethod_name)


# automatically import any Python files in the initmethods directory
initmethods_dir = os.path.dirname(__file__)
import_initmethods(initmethods_dir, "dlk.core.initmethods")
