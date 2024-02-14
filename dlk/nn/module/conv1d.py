# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Collection, Dict, List

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

from dlk.utils.io import open
from dlk.utils.register import register

from . import Module


@cregister("module", "conv1d")
class Conv1dConfig:
    """the 1D convolution config"""

    in_channels = IntField(value=MISSING, minimum=0, help="the input channels")
    out_channels = IntField(value=MISSING, minimum=0, help="the output channels")
    dropout = FloatField(value=0.0, minimum=0.0, maximum=1.0, help="the dropout rate")
    kernel_sizes = ListField(value=[3], help="the kernel sizes")


@register("module", "conv1d")
class Conv1d(Module):
    """Conv for 1d input"""

    def __init__(self, config: Conv1dConfig):
        super().__init__()
        self.config = config
        assert all(
            k % 2 == 1 for k in self.config.kernel_sizes
        ), "the kernel sizes must be odd"
        assert (
            self.config.out_channels % len(self.config.kernel_sizes) == 0
        ), "out channels must be dividable by kernels"
        self.config.out_channels = self.config.out_channels // len(
            self.config.kernel_sizes
        )

        convs = []
        for kernel_size in self.config.kernel_sizes:
            conv = nn.Conv1d(
                self.config.in_channels,
                self.config.out_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            )
            convs.append(nn.Sequential(conv, nn.GELU()))
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(p=float(self.config.dropout))

    def forward(self, x: torch.Tensor):
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns:
            conv result the shape is the same as input

        """
        return self.dropout(torch.cat([conv(x) for conv in self.convs], dim=-1))
