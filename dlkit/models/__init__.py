"""models"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any

MODEL_REGISTRY = {}
MODEL_CONFIG_REGISTRY = {}

def model_config_register(name: str = "") -> Callable:
    """
    register configures
    """
    def decorator(config):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(config.__name__))

        if name in MODEL_CONFIG_REGISTRY:
            raise ValueError('The model config name {} is already registed.'.format(name))
        MODEL_CONFIG_REGISTRY[name] = config
        return config
    return decorator

def model_register(name: str = "") -> Callable:
    """
    register models
    """
    def decorator(model):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(model.__name__))

        if name in MODEL_REGISTRY:
            raise ValueError('The model name {} is already registed.'.format(name))
        MODEL_REGISTRY[name] = model
        return model
    return decorator


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
import_models(models_dir, "dlkit.models")
