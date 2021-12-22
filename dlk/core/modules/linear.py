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


@module_config_register("linear")
class LinearConfig(BaseConfig):
    """Config for Linear

    Config Example:
        >>> {
        >>>     "config": {
        >>>         "input_size": 256,
        >>>         "output_size": 2,
        >>>         "dropout": 0.0, //the module output no need dropout
        >>>         "bias": true, // use bias or not in linear , if set to false, all the bias will be set to 0
        >>>         "pool": null, // pooling output or not
        >>>     },
        >>>     "_name": "linear",
        >>> }
    """
    def __init__(self, config: Dict):
        super(LinearConfig, self).__init__(config)
        config = config['config']
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.dropout = float(config['dropout'])
        self.bias = config['bias']
        self.pool = config['pool']
        self.post_check(config, used=[
            "input_size",
            "output_size",
            "dropout",
            "bias",
            "pool",
        ])


@module_register("linear")
class Linear(Module):
    """wrap for nn.Linear"""
    def __init__(self, config: LinearConfig):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=config.input_size, out_features=config.output_size, )
        self.dropout = nn.Dropout(p=float(config.dropout))
        self.config = config

    def forward(self, input: torch.Tensor)->torch.Tensor:
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns: 
            project result the shape is the same as input(no poll), otherwise depend on poll method

        """
        output = self.dropout(self.linear(input))
        if not self.config.pool:
            return output
        elif self.config.pool == 'first':
            return output[:, 0]
        else:
            raise PermissionError(f"Currenttly we have not support the pool method '{self.config.pool}' in linear.")
