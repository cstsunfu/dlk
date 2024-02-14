# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
import torch.nn as nn
from intc import (
    MISSING,
    AnyField,
    Base,
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

from dlk.utils.register import register


@cregister("initmethod", "range_uniform")
class RangeUniformInitConfig:

    """
    init the parameters by the uniform distribution with the given range
    """

    range_from = FloatField(value=-0.1, help="the lower bound of the init value")
    range_to = FloatField(value=0.1, help="the upper bound of the init value")


@register("initmethod", "range_uniform")
class RangeUniformInit(object):
    """for transformers"""

    def __init__(self, config: RangeUniformInitConfig):
        super().__init__()
        self.config = config

    def __call__(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(
                from_=self.config.range_from, to=self.config.range_to
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(
                from_=self.config.range_from, to=self.config.range_to
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # use the default init,
            # weight are initialized to 1, bias to 0
            module.reset_parameters()
        elif isinstance(module, nn.Conv1d):
            module.weight.data.uniform_(
                from_=self.config.range_from, to=self.config.range_to
            )
        elif isinstance(module, nn.Conv2d):
            module.weight.data.uniform_(
                from_=self.config.range_from, to=self.config.range_to
            )
        elif isinstance(module, nn.Conv3d):
            module.weight.data.uniform_(
                from_=self.config.range_from, to=self.config.range_to
            )
