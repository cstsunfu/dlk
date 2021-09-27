"""decoders"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from utils.config import Config

DECODER_REGISTRY = {}
DECODER_CONFIG_REGISTRY = {}

def decoder_config_register(name: str = "") -> Callable:
    """
    register configures
    """
    def decorator(config):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(config.__name__))

        if name in DECODER_CONFIG_REGISTRY:
            raise ValueError('The decoder config name {} is already registed.'.format(name))
        DECODER_CONFIG_REGISTRY[name] = config
        return config
    return decorator

def decoder_register(name: str = "") -> Callable:
    """
    register decoders
    """
    def decorator(decoder):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(decoder.__name__))

        if name in DECODER_REGISTRY:
            raise ValueError('The decoder name {} is already registed.'.format(name))
        DECODER_REGISTRY[name] = decoder
        return decoder
    return decorator


def import_decoders(decoders_dir, namespace):
    for file in os.listdir(decoders_dir):
        path = os.path.join(decoders_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            decoder_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + decoder_name)


class DecoderInput(object):
    """docstring for DecoderInput"""
    def __init__(self, **args):
        super(DecoderInput, self).__init__()
        self.represent = args.get("represent", None)
        self.input_ids = args.get("input_ids", None)
        self.input_mask = args.get("input_mask", None)


class DecoderOutput(object):
    """docstring for DecoderOutput"""
    def __init__(self, **args):
        super(DecoderOutput, self).__init__()
        self.represent = args.get("represent", None)


# automatically import any Python files in the decoders directory
decoders_dir = os.path.dirname(__file__)
import_decoders(decoders_dir, "decoders")
