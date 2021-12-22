# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        """Process entry

        Args:
            data: 
            >>> {
            >>>     "data": {"train": ...},
            >>>     "tokenizer": ..
            >>> }

        Returns: 
            processed data

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
