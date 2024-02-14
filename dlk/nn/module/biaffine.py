# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.


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

from dlk.utils.register import register

from . import Module


@cregister("module", "biaffine")
class BiAffineConfig:
    """the biaffine module config"""

    input_size = IntField(
        value=MISSING,
        minimum=0,
        help="the input size of the biaffine module",
    )
    hidden_size = IntField(
        value=0,
        minimum=0,
        help="the hidden size of the biaffine module, if set to 0, will set the hidden size to the input size",
    )
    output_size = IntField(
        value=MISSING,
        minimum=0,
        help="the output size of the biaffine module",
    )
    dropout = FloatField(
        value=0.0,
        minimum=0.0,
        maximum=1.0,
        help="the dropout rate of the biaffine module",
    )
    multi_matrix = IntField(
        value=1, minimum=1, help="the number of the matrix of return result"
    )
    max_seq_len = IntField(
        value=1024,
        minimum=0,
        help="the max sequence length of the biaffine module",
    )
    bias = BoolField(value=True, help="whether to use the bias in the biaffine module")
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


@register("module", "biaffine")
class BiAffine(Module):
    """wrap for nn.BiAffine"""

    def __init__(self, config: BiAffineConfig):
        super(BiAffine, self).__init__()
        self.config = config
        if self.config.hidden_size == 0:
            self.config.hidden_size = self.config.input_size
        self.linear_a = nn.Linear(self.config.input_size, self.config.hidden_size)
        self.linear_b = nn.Linear(self.config.input_size, self.config.hidden_size)
        self.dropout = nn.Dropout(p=self.config.dropout)
        if config.active != "none":
            self.active = active_map[config.active]()
        else:
            self.active = lambda x: x
        if self.config.bias:
            self.biaffine = nn.Parameter(
                torch.randn(
                    self.config.hidden_size + 1,
                    self.config.output_size * self.config.multi_matrix,
                    self.config.hidden_size + 1,
                )
            )
        else:
            self.biaffine = nn.Parameter(
                torch.randn(
                    self.config.hidden_size,
                    self.config.output_size * self.config.multi_matrix,
                    self.config.hidden_size,
                )
            )

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

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
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
            output = torch.einsum(
                "bmi,ioj,bnj->bmno",
                torch.cat((input_a, torch.ones_like(input_a[..., :1])), dim=-1),
                self.biaffine,
                torch.cat((input_b, torch.ones_like(input_b[..., :1])), dim=-1),
            )
        else:
            output = torch.einsum(
                "bmi,ioj,bnj->bmno",
                input_a,
                self.biaffine,
                input_b,
            )
        if self.config.multi_matrix > 1:
            bs, seq_len, _, output_size = output.shape
            output = output.reshape(
                bs, seq_len, seq_len, self.config.multi_matrix, self.config.output_size
            )
            output = output.permute(
                0, 3, 1, 2, 4
            )  # bs, group, seq_len, seq_len, output_size)
        return output
