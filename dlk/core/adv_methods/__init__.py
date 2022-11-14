# Copyright cstsunfu. All rights reserved.
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

"""adv_methods"""

import importlib
import os
from typing import Dict, Any
import re
import torch
import torch.nn as nn
from dlk.utils.register import Register
from dlk.utils.logger import Logger

logger = Logger.get_logger()

adv_method_config_register = Register("AdvMethod config register")
adv_method_register = Register("AdvMethod register")

class AdvMethod(object):
    """Save fgm decided by config
    """

    def __init__(self, model: nn.Module, config):
        super().__init__()

    def training_step(self, imodel, batch: Dict[str, torch.Tensor], batch_idx: int):
        """do training_step on a mini batch

        Args:
            imodel: imodel instance
            batch: a mini batch inputs
            batch_idx: the index(dataloader) of the mini batch

        Returns: 
            the outputs

        """
        raise NotImplementedError

def import_adv_methods(adv_methods_dir, namespace):
    for file in os.listdir(adv_methods_dir):
        path = os.path.join(adv_methods_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            adv_method_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + adv_method_name)


# automatically import any Python files in the models directory
adv_methods_dir = os.path.dirname(__file__)
import_adv_methods(adv_methods_dir, "dlk.core.adv_methods")
