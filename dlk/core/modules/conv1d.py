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

import torch.nn as nn
from dlk.utils.config import BaseConfig
import torch
from typing import Dict, List, Collection
from . import Module
from dlk.utils.io import open
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules

@config_register("module", 'conv1d')
@define
class Conv1dConfig(BaseConfig):
    name = NameField(value="conv1d", file=__file__, help="the conv1d config")
    @define
    class Config:
        in_channels = IntField(value="*@*", checker=int_check(lower=0), help="the input channels")
        out_channels = IntField(value="*@*", checker=int_check(lower=0), help="the output channels")
        dropout = FloatField(value=0.0, checker=float_check(lower=0.0), help="the dropout rate")
        kernel_sizes = ListField(value=[3], help="the kernel sizes")
    config = NestField(value=Config, converter=nest_converter)


@register("module", "conv1d")
class Conv1d(Module):
    """Conv for 1d input
    """
    def __init__(self, config: Conv1dConfig):
        super().__init__()
        self.config = config.config
        assert all(k % 2 == 1 for k in self.config.kernel_sizes), 'the kernel sizes must be odd'
        assert self.config.out_channels % len(self.config.kernel_sizes) == 0, 'out channels must be dividable by kernels'
        self.config.out_channels = self.config.out_channels // len(self.config.kernel_sizes)

        convs = []
        for kernel_size in self.config.kernel_sizes:
            conv = nn.Conv1d(self.config.in_channels, self.config.out_channels, kernel_size,
                             padding=(kernel_size - 1) // 2)
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
