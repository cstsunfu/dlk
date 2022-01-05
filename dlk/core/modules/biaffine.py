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
import torch
from typing import Dict, List
from . import module_register, module_config_register, Module
from dlk.utils.config import BaseConfig


@module_config_register("biaffine")
class BiAffineConfig(BaseConfig):
    """Config for BiAffine

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "input_size": 256,
        >>>         "output_size": 2,
        >>>         "dropout": 0.0, //generally no need dropout
        >>>         "bias": true, // use bias or not in biaffine
        >>>     },
        >>>     "_name": "biaffine",
        >>> }
    """
    def __init__(self, config: Dict):
        super(BiAffineConfig, self).__init__(config)
        config = config['config']
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.dropout = float(config['dropout'])
        self.bias = config['bias']
        self.post_check(config, used=[
            "input_size",
            "output_size",
            "dropout",
            "bias",
        ])


@module_register("biaffine")
class BiAffine(Module):
    """wrap for nn.BiAffine"""
    def __init__(self, config: BiAffineConfig):
        super(BiAffine, self).__init__()
        if config.bias:
            self.biaffine = nn.Parameter(torch.randn(config.input_size+1, config.output_size, config.input_size+1))
        else:
            self.biaffine = nn.Parameter(torch.randn(config.input_size, config.output_size, config.input_size))

        self.dropout = nn.Dropout(p=float(config.dropout))
        self.config = config

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns: 
            None

        """
        torch.nn.init.xavier_uniform_(self.biaffine)

    def forward(self, input_a: torch.Tensor, input_b: torch.Tensor)->torch.Tensor:
        """do forward on a mini batch

        Args:
            input_a: a mini batch inputs_a, shape==(batch_size, input_a_len, input_size)
            input_b: a mini batch inputs_b, shape==(batch_size, input_b_len, input_size)

        Returns: 
            input_a x biaffine x input_b, shape==(batch_size, input_a_len, input_b_len, output_size)

        """
        if self.config.bias:
            output = self.dropout(torch.einsum('bmi,ioj,bnj->bmno', 
                    torch.cat((input_a, torch.ones_like(input_a[..., :1])), dim=-1), 
                    self.biaffine, 
                    torch.cat((input_b, torch.ones_like(input_b[..., :1])), dim=-1)
                    ))
        else:
            output = self.dropout(torch.einsum('bmi,ioj,bnj->bmno', 
                    input_a,
                    self.biaffine, 
                    input_b,
                    ))
        return output
