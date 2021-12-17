"""basic modules"""
import importlib
import os
from dlk.utils.register import Register

module_config_register = Register("Module config register.")
module_register = Register("Module register.")


def import_modules(modules_dir, namespace):
    for file in os.listdir(modules_dir):
        path = os.path.join(modules_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            module_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + module_name)


# automatically import any Python files in the modules directory
modules_dir = os.path.dirname(__file__)
import_modules(modules_dir, "dlk.core.modules")
