# Register  all
"""processors"""

import importlib
import os
from typing import Callable, Dict, Type
from dlkit.utils.config import Config
from dlkit.utils.register import Register
import abc

subprocessor_config_register = Register("Processor config register")
subprocessor_register = Register("Processor register")


class Processor(metaclass=abc.ABCMeta):
    """docstring for Processor"""
    @abc.abstractmethod
    def process(self, data: Dict)->Dict:
        """TODO: Docstring for process.

        :arg1: TODO
        :returns: TODO

        """
        raise NotImplementedError


def import_subprocessors(processors_dir, namespace):
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
subprocessors_dir = os.path.dirname(__file__)
import_subprocessors(subprocessors_dir, "dlkit.processors.subprocessors")
