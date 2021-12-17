"""models"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlk.utils.register import Register

model_config_register = Register("Model config register.")
model_register = Register("Model register.")

def import_models(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + model_name)


# automatically import any Python files in the models directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "dlk.core.models")
