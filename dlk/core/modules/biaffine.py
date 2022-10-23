# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:# www.apache.org/licenses/LICENSE-2.0
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
    default_config = {
        "_name": "biaffine",
        "config": {
            "input_size": "*@*",
            "hidden_size": 0, # default == input_size
            "output_size": "*@*",
            "dropout": 0.0, # generally no need dropout
            "multi_matrix": 1, # like relation need head pair and tail pair calc togather, so the multi_matrix should set to >1
            "relation_position": False, # whether add relation_position before align
            "bias": True, # use bias or not in biaffine
        },
    }
    """Config for BiAffine
    Config Example:
    """
    def __init__(self, config: Dict):
        super(BiAffineConfig, self).__init__(config)
        config = config['config']
        self.input_size = config['input_size']
        self.multi_matrix = config['multi_matrix']
        self.relation_position = config['relation_position']
        self.target_size = config['output_size']
        self.output_size = config['output_size'] * self.multi_matrix
        self.hidden_size = config['hidden_size']
        if not self.hidden_size:
            self.hidden_size = self.input_size
        self.dropout = config['dropout']
        self.dropout = float(config['dropout'])
        self.bias = config['bias']
        self.post_check(config, used=[
            "input_size",
            "hidden_size",
            "multi_matrix",
            "output_size",
            "dropout",
            "bias",
        ])


@module_register("biaffine")
class BiAffine(Module):
    """wrap for nn.BiAffine"""
    def __init__(self, config: BiAffineConfig):
        super(BiAffine, self).__init__()
        self.linear_a = nn.Linear(config.input_size, config.hidden_size)
        self.linear_b = nn.Linear(config.input_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.dropout)
        self.active = nn.LeakyReLU() # TODO: why GELU get loss nan?
        if config.bias:
            self.biaffine = nn.Parameter(torch.randn(config.hidden_size+1, config.output_size, config.hidden_size+1))
        else:
            self.biaffine = nn.Parameter(torch.randn(config.hidden_size, config.output_size, config.hidden_size))

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

    def forward(self, embedding: torch.Tensor)->torch.Tensor:
        """do forward on a mini batch

        Args:
            embedding: a mini batch embedding, shape==(batch_size, input_a_len, input_size)

        Returns: 
            input_a x biaffine x input_b, shape==(batch_size, input_a_len, input_b_len, output_size)

        """

        input_a = self.dropout(self.active(embedding))
        input_b = self.dropout(self.active(embedding))
        input_a = self.linear_a(input_a)
        input_b = self.linear_b(input_b)
        if self.config.bias:
            output = torch.einsum('bmi,ioj,bnj->bmno', 
                    torch.cat((input_a, torch.ones_like(input_a[..., :1])), dim=-1), 
                    self.biaffine, 
                    torch.cat((input_b, torch.ones_like(input_b[..., :1])), dim=-1)
                    )
        else:
            output = torch.einsum('bmi,ioj,bnj->bmno', 
                    input_a,
                    self.biaffine, 
                    input_b,
                    )
        if self.config.multi_matrix>1:
            bs, seq_len, _, output_size = output.shape
            output = output.reshape(bs, seq_len, seq_len, self.config.multi_matrix, self.config.target_size)
            output = output.permute(0, 3, 1, 2, 4) # bs, group, seq_len, seq_len, target_size)
        return output
