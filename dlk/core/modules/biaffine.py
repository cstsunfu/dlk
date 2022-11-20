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
import numpy as np
import torch
from typing import Dict, List, Optional
from . import module_register, module_config_register, Module
from dlk.utils.config import BaseConfig
from dlk.utils.logger import Logger

logger = Logger.get_logger()

class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer

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
            "max_seq_len": 1024, # for relation_position
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
        self.max_seq_len = config['max_seq_len']
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
            "output_size",
            "dropout",
            "multi_matrix",
            "relation_position",
            "max_seq_len",
            "bias",
        ])


@module_register("biaffine")
class BiAffine(Module):
    """wrap for nn.BiAffine"""
    def __init__(self, config: BiAffineConfig):
        super(BiAffine, self).__init__()
        self.linear_a = nn.Linear(config.input_size, config.hidden_size)
        self.linear_b = nn.Linear(config.input_size, config.hidden_size)
        self.config = config
        if config.relation_position:
            self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
                config.max_seq_len, config.hidden_size
            )
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
        self.linear_a.apply(method)
        self.linear_b.apply(method)

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

        if self.config.relation_position:
            sinusoidal_pos = self.embed_positions(input_a.shape[:-1])[None, :, :]
            input_a, input_b = self.embed_positions.apply_rotary_position_embeddings(sinusoidal_pos, input_a, input_b)

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
