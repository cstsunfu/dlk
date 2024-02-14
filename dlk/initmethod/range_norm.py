# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

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


@cregister("initmethod", "range_norm")
class RangeNormInitConfig:
    """
    init the parameters by the normal distribution with the given std
    """

    std = FloatField(
        value=0.1,
        minimum=0.0,
        help="the std of the normal distribution init method",
    )


@register("initmethod", "range_norm")
class RangeNormInit(object):
    """default for transformers init method"""

    def __init__(self, config: RangeNormInitConfig):
        super().__init__()
        self.std = config.std

    def __call__(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # use the default init
            # weight are initialized to 1, bias to 0
            module.reset_parameters()
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.std)
        elif isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.std)
        elif isinstance(module, nn.Conv3d):
            module.weight.data.normal_(mean=0.0, std=self.std)
