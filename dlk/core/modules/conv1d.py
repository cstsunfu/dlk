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
from . import module_register, module_config_register, Module

@module_config_register("conv1d")
class Conv1dConfig(BaseConfig):
    """Config for Conv1d

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "in_channels": "*@*",
        >>>         "out_channels": "*@*",
        >>>         "dropout": 0.0,
        >>>         "kernel_sizes": [3],
        >>>     },
        >>>     "_name": "conv1d",
        >>> }
    """
    def __init__(self, config: Dict):
        super(Conv1dConfig, self).__init__(config)
        config = config['config']
        self.kernel_sizes = config['kernel_sizes']
        out_channels = config['out_channels']
        assert all(k % 2 == 1 for k in self.kernel_sizes), 'the kernel sizes must be odd'
        assert out_channels % len(self.kernel_sizes) == 0, 'out channels must be dividable by kernels'
        self.in_channels = config['in_channels']
        self.out_channels = out_channels // len(self.kernel_sizes)
        self.dropout = config['dropout']
        self.post_check(config, used=[
            "in_channels",
            "out_channels",
            "dropout",
            "kernel_sizes",
        ])

@module_register("conv1d")
class Conv1d(Module):
    """Conv for 1d input
    """
    def __init__(self, config: Conv1dConfig):
        super().__init__()
        convs = []
        for kernel_size in config.kernel_sizes:
            conv = nn.Conv1d(config.in_channels, config.out_channels, kernel_size,
                             padding=(kernel_size - 1) // 2)
            convs.append(nn.Sequential(conv, nn.GELU()))
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(p=float(config.dropout))

    def forward(self, x: torch.Tensor):
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: 
            conv result the shape is the same as input

        """
        return self.dropout(torch.cat([conv(x) for conv in self.convs], dim=-1))
