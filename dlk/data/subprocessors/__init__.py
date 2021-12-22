#Paras Copyright 2021 cstsunfu. All rights reserved.
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

# Register  all
"""processors"""

import importlib
import os
from typing import Callable, Dict, Type
from dlk.utils.config import ConfigTool
from dlk.utils.register import Register
import abc
from pandarallel import pandarallel
pandarallel.initialize(verbose=1)
# TODO: tokenizer parall and other parall is anti
os.environ["TOKENIZERS_PARALLELISM"] = "false"

subprocessor_config_register = Register("SubProcessor config register")
subprocessor_register = Register("SubProcessor register")

class ISubProcessor(metaclass=abc.ABCMeta):
    """docstring for ISubProcessor"""

    @abc.abstractmethod
    def process(self, data: Dict)->Dict:
        """SubProcess entry

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
import_subprocessors(subprocessors_dir, "dlk.data.subprocessors")
