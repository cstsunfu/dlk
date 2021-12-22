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

"""basic modules"""
import importlib
import os
from dlk.utils.register import Register
import torch.nn as nn
from typing import Dict
import torch

module_config_register = Register("Module config register.")
module_register = Register("Module register.")


class Module(nn.Module):
    """This class is means DLK Module for replace the torch.nn.Module in this project
    """

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        for module in self.children():
            module.apply(method)

    def forward(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """in simple module, all step fit to this method

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        raise NotImplementedError

    def predict_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do predict for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs)

    def training_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do train for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs)

    def validation_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do validation for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs)

    def test_step(self, inputs: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        """do test for one batch

        Args:
            inputs: one mini-batch inputs

        Returns: 
            one mini-batch outputs

        """
        return self(inputs)



def import_modules(modules_dir, namespace):
    for file in os.listdir(modules_dir):
        path = os.path.join(modules_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            module_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + module_name)


# automatically import any Python files in the modules directory
modules_dir = os.path.dirname(__file__)
import_modules(modules_dir, "dlk.core.modules")
