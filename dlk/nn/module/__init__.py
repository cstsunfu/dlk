# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict

import torch
import torch.nn as nn
from intc import (
    AnyField,
    BoolField,
    DictField,
    FloatField,
    IntField,
    ListField,
    NestField,
    StrField,
    SubModule,
    cregister,
)

from dlk.nn.base_module import BaseIdentityModule
from dlk.utils.import_module import import_module_dir
from dlk.utils.register import register


@cregister("module", "identity")
class IdentityModuleConfig:
    """identity module"""

    pass


@register("module", "identity")
class IdentityModule(BaseIdentityModule):
    pass


class Module(nn.Module):
    """This class is means DLK Module for replace the torch.nn.Module in this project"""

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        for module in self.children():
            module.apply(method)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """in simple module, all step fit to this method

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        raise NotImplementedError

    def predict_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do predict for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        return self(inputs)

    def training_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do train for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        return self(inputs)

    def validation_step(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """do validation for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        return self(inputs)

    def test_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """do test for one batch

        Args:
            inputs: one mini-batch inputs

        Returns:
            one mini-batch outputs

        """
        return self(inputs)


module_dir = os.path.dirname(__file__)
import_module_dir(module_dir, "dlk.nn.module")
