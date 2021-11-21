"""postprocessors"""

import importlib
import os
from typing import Callable, Dict, Type
from dlkit.utils.register import Register
import abc

class IPostProcessor(metaclass=abc.ABCMeta):
    """docstring for IPostProcessor"""


    @abc.abstractmethod
    def process(self, data: Dict)->Dict:
        """TODO: Docstring for process.

        :arg1: TODO
        :returns: TODO

        """
        raise NotImplementedError
        

postprocessor_config_register = Register('PostProcessor config register')
postprocessor_register = Register("PostProcessor register")


def import_postprocessors(postprocessors_dir, namespace):
    for file in os.listdir(postprocessors_dir):
        path = os.path.join(postprocessors_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            # and not (file.endswith("subpostprocessors") and os.path.isdir(path))
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            postprocessor_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + postprocessor_name)


# automatically import any Python files in the models directory
postprocessors_dir = os.path.dirname(__file__)
import_postprocessors(postprocessors_dir, "dlkit.postprocessors")
