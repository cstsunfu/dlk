"""processors"""

import importlib
import os
from typing import Callable, Dict, Type
from dlkit.utils.config import Config
import abc

class Processor(metaclass=abc.ABCMeta):
    """docstring for Processor"""


    @abc.abstractmethod
    def process(self, data: Dict)->Dict:
        """TODO: Docstring for process.

        :arg1: TODO
        :returns: TODO

        """
        raise NotImplementedError
        

PROCESSOR_REGISTRY: Dict[str, Processor] = {}
PROCESSOR_CONFIG_REGISTRY: Dict[str, Config] = {}


def processor_config_register(name: str = "") -> Callable:
    """
    register configures
    """
    def decorator(config):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(config.__name__))

        if name in PROCESSOR_CONFIG_REGISTRY:
            raise ValueError('The processor config name {} is already registed.'.format(name))
        PROCESSOR_CONFIG_REGISTRY[name] = config
        return config
    return decorator

def processor_register(name: str = "") -> Callable:
    """
    register processors
    """
    def decorator(processor):
        if name.strip() == "":
            raise ValueError('You must set a name for {}'.format(processor.__name__))

        if name in PROCESSOR_REGISTRY:
            raise ValueError('The processor name {} is already registed.'.format(name))
        PROCESSOR_REGISTRY[name] = processor
        return processor
    return decorator


def import_processors(processors_dir, namespace):
    for file in os.listdir(processors_dir):
        path = os.path.join(processors_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            processor_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + processor_name)


# automatically import any Python files in the models directory
processors_dir = os.path.dirname(__file__)
import_processors(processors_dir, "dlkit.processors")
