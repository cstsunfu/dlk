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
from dlk.nn.utils.rope import RoFormerSinusoidalPositionalEmbedding
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
        help="the hidden size of the bilinear module, if set to 0, will set the hidden size to the input size",
    )
    output_size = IntField(
        value=MISSING,
        minimum=0,
        help="the output size of the bilinear module",
    )
    dropout = FloatField(
        value=0.0,
        minimum=0.0,
        maximum=1.0,
        help="the dropout rate of the bilinear module",
    )
    max_seq_len = IntField(
        value=1024,
        minimum=0,
        help="the max sequence length of the bilinear module",
    )
    relation_position = BoolField(
        value=False, help="whether to use the relative position"
    )
    active = StrField(
        value="leaky_relu",
        options=["none", "leaky_relu", "relu", "gelu", "glu", "selu", "celu", "elu"],
        help="the activation function",
    )


active_map = {
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "glu": nn.GLU,
    "selu": nn.SELU,
    "celu": nn.CELU,
    "elu": nn.ELU,
}


@register("module", "bilinear")
class BiLinear(Module):
    """"""

    def __init__(self, config: BiLinearConfig):
        super(BiLinear, self).__init__()
        self.config = config
        if self.config.hidden_size == 0:
            self.config.hidden_size = self.config.input_size
        self.linear_a = nn.Linear(self.config.input_size, self.config.hidden_size)
        self.linear_b = nn.Linear(self.config.input_size, self.config.hidden_size)
        if self.config.relation_position:
            self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
                self.config.max_seq_len, self.config.hidden_size
            )
        self.dropout = nn.Dropout(p=self.config.dropout)
        if config.active != "none":
            self.active = active_map[config.active]()
        else:
            self.active = lambda x: x

    def init_weight(self, method):
        """init the weight of submodules by 'method'

        Args:
            method: init method

        Returns:
            None

        """
        self.linear_a.apply(method)
        self.linear_b.apply(method)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """do forward on a mini batch

        Args:
            embedding: a mini batch embedding, shape==(batch_size, input_a_len, input_size)

        Returns:
            input_a x bilinear x input_b, shape==(batch_size, input_a_len, input_b_len, output_size)

        """

        input_a = self.dropout(self.active(embedding))
        input_b = self.dropout(self.active(embedding))
        input_a = self.linear_a(input_a)
        input_b = self.linear_b(input_b)

        if self.config.relation_position:
            sinusoidal_pos = self.embed_positions(input_a.shape[:-1])[None, :, :]
            input_a, input_b = self.embed_positions.apply_rotary_position_embeddings(
                sinusoidal_pos, input_a, input_b
            )

        return output
