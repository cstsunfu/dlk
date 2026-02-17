# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import numpy as np
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

from dlk.nn.module import Module
from dlk.nn.utils.rope import (
    RoFormerSinusoidalPositionalEmbedding,
)  # Assuming this is in a local file
from dlk.utils.register import register


@cregister("module", "bilinear")
class BiLinearConfig:
    """the bilinear module config"""

    input_size = IntField(
        value=MISSING,
        minimum=0,
        help="the input size of the bilinear module",
    )
    hidden_size = IntField(
        value=0,
        minimum=0,
        help="the hidden size of the bilinear module (equivalent to inner_dim in GlobalPointer), if set to 0, will set the hidden size to the input size",
    )
    output_size = IntField(
        value=MISSING,
        minimum=0,
        help="the output size of the bilinear module (equivalent to ent_type_size in GlobalPointer)",
    )
    max_seq_len = IntField(
        value=1024,
        minimum=0,
        help="the max sequence length of the bilinear module, used for RoPE",
    )
    relation_position = BoolField(
        value=False, help="whether to use the relative position (RoPE)"
    )
    efficient = BoolField(
        value=False,
        help="If True, use efficient bilinear like EffiGlobalPointer.",
    )


@register("module", "bilinear")
class BiLinear(Module):
    """
    A general Bilinear module with optional Rotary Position Embeddings (RoPE),
    inspired by RawGlobalPointer and EffiGlobalPointer.

    Args:
        config (BiLinearConfig): Configuration object for the module.
    """

    def __init__(self, config: BiLinearConfig):
        super(BiLinear, self).__init__()
        self.config = config
        if self.config.hidden_size == 0:
            self.config.hidden_size = self.config.input_size

        if not self.config.efficient:
            # RawGlobalPointer style: one large linear layer
            self.dense = nn.Linear(
                self.config.input_size,
                self.config.output_size * self.config.hidden_size * 2,
                bias=False,
            )
        else:
            self.dense1 = nn.Linear(
                self.config.input_size, self.config.hidden_size * 2, bias=False
            )
            self.dense2 = nn.Linear(
                self.config.input_size, self.config.output_size * 2, bias=False
            )

        if self.config.relation_position:
            self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
                self.config.max_seq_len, self.config.hidden_size
            )

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method
        """
        if hasattr(self, "dense"):
            self.dense.apply(method)
        if hasattr(self, "dense1"):
            self.dense1.apply(method)
        if hasattr(self, "dense2"):
            self.dense2.apply(method)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """do forward on a mini batch

        Args:
            embedding: a mini batch embedding, shape==(batch_size, seq_len, input_size)

        Returns:
            Logits tensor, shape==(batch_size, output_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = embedding.shape

        if not self.config.efficient:
            # RawGlobalPointer style
            # (b, s, i) -> (b, s, o * h * 2)
            outputs = self.dense(embedding)
            # -> List[(b, s, h * 2)] with len o -> (b, s, o, h * 2)
            outputs = torch.stack(
                torch.split(outputs, self.config.hidden_size * 2, dim=-1), dim=-2
            )
            # -> (b, s, o, h), (b, s, o, h)
            qw, kw = (
                outputs[..., : self.config.hidden_size],
                outputs[..., self.config.hidden_size :],
            )
            qw = qw.permute(0, 2, 1, 3).contiguous()
            kw = kw.permute(0, 2, 1, 3).contiguous()

            if self.config.relation_position:
                # (s, h) -> (1, 1, s, h) for broadcasting
                sinusoidal_pos = self.embed_positions(seq_len).unsqueeze(0).unsqueeze(0)
                qw, kw = self.embed_positions.apply_rotary_position_embeddings(
                    sinusoidal_pos, qw, kw
                )

            logits = torch.einsum("bosh,bokh->bosk", qw, kw)

        else:
            # (b, s, i) -> (b, s, h * 2)
            outputs = self.dense1(embedding)
            # -> (b, s, h), (b, s, h)
            qw, kw = outputs[..., ::2].contiguous(), outputs[..., 1::2].contiguous()

            if self.config.relation_position:
                # (s, h) -> (1, s, h) for broadcasting
                sinusoidal_pos = self.embed_positions(seq_len).unsqueeze(0)
                qw, kw = self.embed_positions.apply_rotary_position_embeddings(
                    sinusoidal_pos, qw, kw
                )

            # qw: (b, s, h), kw: (b, s, h) -> kw.T: (b, h, s)
            logits = torch.matmul(qw, kw.transpose(-1, -2))

            # (b, s, i) -> (b, s, o * 2)
            bias = self.dense2(embedding)
            # -> (b, s, o, 2) -> (b, o, s, 2)
            bias = bias.view(batch_size, seq_len, self.config.output_size, 2).permute(
                0, 2, 1, 3
            )
            # -> (b, o, s), (b, o, s)
            bias_start, bias_end = bias[..., 0], bias[..., 1]

            # Scale logits and bias
            logits = logits / self.config.hidden_size**0.5
            bias = (bias_start.unsqueeze(3) + bias_end.unsqueeze(2)) / 2

            # Add bias via broadcasting
            # (b, 1, s, s) + (b, o, s, 1) + (b, o, 1, s) -> (b, o, s, s)
            logits = logits.unsqueeze(1) + bias

        return logits
