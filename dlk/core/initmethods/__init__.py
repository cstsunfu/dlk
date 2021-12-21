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

"""initmethods"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlk.utils.register import Register

initmethod_config_register = Register("InitMethod config register.")
initmethod_register = Register("InitMethod register.")


def import_initmethods(initmethods_dir, namespace):
    for file in os.listdir(initmethods_dir):
        path = os.path.join(initmethods_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            initmethod_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + initmethod_name)


# automatically import any Python files in the initmethods directory
initmethods_dir = os.path.dirname(__file__)
import_initmethods(initmethods_dir, "dlk.core.initmethods")
