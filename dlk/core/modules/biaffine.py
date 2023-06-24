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
from . import Module
from dlk import register, config_register
from dlk.utils.config import define, float_check, int_check, str_check, number_check, options, suggestions, nest_converter
from dlk.utils.config import BaseConfig, IntField, BoolField, FloatField, StrField, NameField, AnyField, NestField, ListField, DictField, NumberField, SubModules

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

@config_register("module", 'biaffine')
@define
class BiAffineConfig(BaseConfig):
    name = NameField(value="biaffine", file=__file__, help="the biaffine module config")
    @define
    class Config:
        input_size = IntField(value="*@*", checker=int_check(lower=0), help="the input size of the biaffine module")
        hidden_size = IntField(value=0, checker=int_check(lower=0), help="the hidden size of the biaffine module, if set to 0, will set the hidden size to the input size")
        output_size = IntField(value="*@*", checker=int_check(lower=0), help="the output size of the biaffine module")
        dropout = FloatField(value=0.0, checker=float_check(lower=0.0, upper=1.0), help="the dropout rate of the biaffine module")
        multi_matrix = IntField(value=1, checker=int_check(lower=1), help="the number of the matrix")
        max_seq_len = IntField(value=1024, checker=int_check(lower=0), help="the max sequence length of the biaffine module")
        relation_position = BoolField(value=False, help="whether to use the relative position")
        bias = BoolField(value=True, help="whether to use the bias in the biaffine module")
    config = NestField(value=Config, converter=nest_converter)


@register("module", "biaffine")
class BiAffine(Module):
    """wrap for nn.BiAffine"""
    def __init__(self, config: BiAffineConfig):
        super(BiAffine, self).__init__()
        self.config = config.config
        if self.config.hidden_size == 0:
            self.config.hidden_size = self.config.input_size
        self.linear_a = nn.Linear(self.config.input_size, self.config.hidden_size)
        self.linear_b = nn.Linear(self.config.input_size, self.config.hidden_size)
        if self.config.relation_position:
            self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
                self.config.max_seq_len, self.config.hidden_size
            )
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.active = nn.LeakyReLU() # TODO: why GELU get loss nan?
        if self.config.bias:
            self.biaffine = nn.Parameter(torch.randn(self.config.hidden_size+1, self.config.output_size, self.config.hidden_size+1))
        else:
            self.biaffine = nn.Parameter(torch.randn(self.config.hidden_size, self.config.output_size, self.config.hidden_size))

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
            output = output.reshape(bs, seq_len, seq_len, self.config.multi_matrix, self.config.output_size)
            output = output.permute(0, 3, 1, 2, 4) # bs, group, seq_len, seq_len, output_size)
        return output
