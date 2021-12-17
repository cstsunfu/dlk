"""processors"""

import importlib
import os
from typing import Callable, Dict, Type
from dlk.utils.register import Register
import abc

class IProcessor(metaclass=abc.ABCMeta):
    """docstring for IProcessor"""


    @abc.abstractmethod
    def process(self, data: Dict)->Dict:
        """TODO: Docstring for process.

        :arg1: TODO
        :returns: TODO

        """
        raise NotImplementedError


processor_config_register = Register('Processor config register')
processor_register = Register("Processor register")


def import_processors(processors_dir, namespace):
    for file in os.listdir(processors_dir):
        path = os.path.join(processors_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            # and not (file.endswith("subprocessors") and os.path.isdir(path))
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            processor_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + processor_name)


# automatically import any Python files in the models directory
processors_dir = os.path.dirname(__file__)
import_processors(processors_dir, "dlk.data.processors")
