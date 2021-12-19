import torch.nn as nn
from dlk.utils.config import BaseConfig
import torch
from typing import Dict, List, Collection
from . import module_register, module_config_register

@module_config_register("conv1d")
class Conv1dConfig(BaseConfig):
    """docstring for Conv1dConfig
    {
        "config": {
            "in_channels": "*@*",
            "out_channels": "*@*",
            "dropout": 0.0,
            "kernel_sizes": [3],
        },
        "_name": "conv1d",
    }
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
class Conv1d(nn.Module):
    def __init__(self, config: Conv1dConfig):
        super().__init__()
        convs = []
        for kernel_size in config.kernel_sizes:
            conv = nn.Conv1d(config.in_channels, config.out_channels, kernel_size,
                             padding=(kernel_size - 1) // 2)
            convs.append(nn.Sequential(conv, nn.GELU()))
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(p=float(config.dropout))

    def forward(self, x):
        return self.dropout(torch.cat([conv(x) for conv in self.convs], dim=-1))
