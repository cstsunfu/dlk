"""encoders"""

import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlkit.utils.config import Config

ENCODER_REGISTRY = {}
ENCODER_CONFIG_REGISTRY = {}

def encoder_config_register(name: str = "") -> Callable:
    """
    register configures
    """
    def decorator(config):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(config.__name__))

        if name in ENCODER_CONFIG_REGISTRY:
            raise ValueError('The encoder config name {} is already registed.'.format(name))
        ENCODER_CONFIG_REGISTRY[name] = config
        return config
    return decorator

def encoder_register(name: str = "") -> Callable:
    """
    register encoders
    """
    def decorator(encoder):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(encoder.__name__))

        if name in ENCODER_REGISTRY:
            raise ValueError('The encoder name {} is already registed.'.format(name))
        ENCODER_REGISTRY[name] = encoder
        return encoder
    return decorator


def import_encoders(encoders_dir, namespace):
    for file in os.listdir(encoders_dir):
        path = os.path.join(encoders_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            encoder_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + encoder_name)


class EncoderInput(object):
    """docstring for EncoderInput"""
    def __init__(self, **args):
        super(EncoderInput, self).__init__()
        self.represent = args.get("represent", None)
        self.input_ids = args.get("input_ids", None)
        self.input_mask = args.get("input_mask", None)


class EncoderOutput(object):
    """docstring for EncoderOutput"""
    def __init__(self, **args):
        super(EncoderOutput, self).__init__()
        self.represent = args.get("represent", None)


# automatically import any Python files in the encoders directory
encoders_dir = os.path.dirname(__file__)
import_encoders(encoders_dir, "encoders")
