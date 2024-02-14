# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

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


@cregister("module", "linear")
class LinearConfig:
    """
    The Linear module config
    """

    input_size = IntField(value=MISSING, minimum=0, help="the input size")
    output_size = IntField(value=MISSING, minimum=0, help="the output size")
    dropout = FloatField(value=0.0, minimum=0.0, maximum=1.0, help="the dropout rate")
    bias = BoolField(value=True, help="whether to use bias")
    pool = StrField(
        value=None,
        options=["first", None],
        help="the pooling method, currently only implement `first`",
    )


@register("module", "linear")
class Linear(Module):
    """wrap for nn.Linear"""

    def __init__(self, config: LinearConfig):
        super(Linear, self).__init__()
        self.config = config
        self.linear = nn.Linear(
            in_features=self.config.input_size, out_features=self.config.output_size
        )
        self.dropout = nn.Dropout(p=float(self.config.dropout))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """do forward on a mini batch

        Args:
            batch: a mini batch inputs

        Returns:
            project result the shape is the same as input(no poll), otherwise depend on poll method

        """
        output = self.dropout(self.linear(input))
        if not self.config.pool:
            return output
        elif self.config.pool == "first":
            return output[:, 0]
        else:
            raise PermissionError(
                f"Currenttly we have not support the pool method '{self.config.pool}' in linear."
            )
